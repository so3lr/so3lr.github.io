# Enhanced sampling

SO3LR integrates with [PySAGES](https://github.com/SSAGESLabs/PySAGES), a
JAX-based enhanced-sampling library, so you can run biased MD with collective
variables. This is useful for free-energy calculations and for accelerated
exploration of configuration space.

## Installation

PySAGES is an optional dependency. To install it from a repository checkout:

```shell
uv sync --locked --extra pysages
```

To install it into an existing environment, use `pip install ".[pysages]"`.
The extra installs PySAGES from a pinned fork and commit, and we recommend it
over a separately installed PySAGES.

Enhanced-sampling code runs only when `do_enhanced_sampling: true` is set, and
always on the neighbour-list engine (`engine: flash` is refused). See
[MD engines](../usage/md.md#md-engines).

## Methods

| Method | Description |
|---|---|
| `unbiased` | Unbiased MD with CV tracking, useful as a baseline |
| `abf` | Adaptive Biasing Force |
| `spectralabf` | Spectral ABF, with spectral decomposition of the bias |
| `metad` | Well-tempered metadynamics |
| `harmonic` | Harmonic bias (umbrella sampling, string method) |

## Collective variables

| CV | Atom keys | Description |
|---|---|---|
| `distance` | `grp1`, `grp2` | Distance between two atom groups |
| `differenceofdistances` | `indices` | Difference of two pairwise distances |
| `meanofdistances` | `indices` | Mean of pairwise distances |
| `angle` | `indices` | Bond angle (3 atoms) |
| `dihedral` | `indices` | Dihedral angle (4 atoms) |
| `gyrrad` | `indices` | Radius of gyration |
| `princmom` | `indices`, `axis` | Principal moment of inertia |
| `component` | `indices`, `axis` | Cartesian component (`axis` 0/1/2 = x/y/z) of a group's barycentre |
| `relativecomponent` | `grp1`, `grp2`, `axis` | Same, relative to a second group |
| `asphericity` | `indices` | Asphericity |
| `acylindricity` | `indices` | Acylindricity |
| `shapeanisotropy` | `indices` | Shape anisotropy |
| `coordinationnumber` | `indices` | Coordination number, with element filtering (`species`, `cn_exps`, `indices_nn`) |
| `ringphase`, `ringamplitude` | `indices` | Ring pucker phase angle and amplitude |

CV type names are case-insensitive. The minimum-image variants `distance_pbc`,
`differenceofdistances_pbc`, `meanofdistances_pbc` and `relativecomponent_pbc`
take the same keys and require a periodic box.

Atom indices are **0-based** in the input geometry's order, given as plain
integer lists. Every CV except the two-group ones above takes a single flat
`indices` list, so a `dihedral` is `indices: [4, 6, 8, 14]`. The `grp1` and
`grp2` of `relativecomponent` and `relativecomponent_pbc` also accept
inclusive ranges, such as `grp2: "0-19"` or `grp1: [0, "3-5"]`.

## Supported ensembles

`abf`, `spectralabf` and `metad` require NVT (Nosé–Hoover, `ensemble: "nvt"`).
`harmonic` and `unbiased` also run under `npt` and `nve`. `nvt-langevin` is
not supported by any method.

## Configuration

Enhanced sampling uses two files, the usual MD settings file and a PySAGES
file that defines the CVs and the method.

```yaml title="md_settings.yaml"
ensemble: "nvt"
input_file: "geometry.xyz"
output_file: "trajectory.xyz"
md_dt: 0.5
md_T: 300
md_cycles: 100
md_steps: 100

do_enhanced_sampling: true
pysages_input: "./pysages_input.yaml"      # default: ./pysages_input.yaml
restart_pysages_save_path: "./restart_pysages.pkl"
```

=== "ABF"

    ```yaml title="pysages_input.yaml"
    method: "abf"

    cv:
      - type: "distance"
        grp1: [0]
        grp2: [1]
        grid_min: 1.0
        grid_max: 5.0
        grid_stride: 50
        periodic: false

    # Optional: restraints keeping the CV inside the grid
    restraints:
      lower: [1.5]
      upper: [4.5]
      kl: [50.0]
      ku: [50.0]
    ```

=== "Harmonic bias"

    ```yaml title="pysages_input.yaml"
    method: "harmonic"

    cv:
      - type: "distance"
        grp1: [0]
        grp2: [1]
        grid_min: 1.0
        grid_max: 5.0
        grid_stride: 50
        periodic: false

    method_args:
      kspring: [100.0]
      center: [3.0]
    ```

=== "Metadynamics"

    ```yaml title="pysages_input.yaml"
    method: "metad"

    cv:
      - type: "dihedral"
        indices: [4, 6, 8, 14]
        grid_min: -3.14159
        grid_max: 3.14159
        grid_stride: 64
        periodic: true

    # all five keys are required for metad
    method_args:
      height: 0.1        # eV
      sigma: [0.2]       # one width per CV
      stride: 100        # steps between deposited Gaussians
      ngauss: 500        # maximum number of Gaussians
      deltaT: 1500.0     # well-tempered ΔT in K
    ```

Then run:

```shell
so3lr --settings md_settings.yaml
```

The bias acts within each MD cycle, and its state carries over from one cycle
to the next. Energies are in eV, distances in Å and angles in radians.

Every CV needs `grid_min`, `grid_max`, `grid_stride` and `periodic`, whatever
the method. `abf` and `spectralabf` take optional `method_args`: `N` (default
`500`) and `use_pinv` (default `false`). `harmonic` requires `kspring` and
`center`, and `metad` requires all five keys above. With several CVs of mixed
periodicity (say a dihedral plus a distance), the grid is treated as
non-periodic.

## Restarting

The PySAGES state (the accumulated bias, histograms and estimators) is saved
to its own `.pkl` file next to the SO3LR `.npz` checkpoint, every
`save_buffer` cycles and at the end of the run.

```yaml
# First run: save both
restart_save_path: "./restart_so3lr.npz"
restart_pysages_save_path: "./restart_pysages.pkl"

# Continuation: load both, save to fresh files
restart_load_path: "./restart_so3lr.npz"
restart_pysages_load_path: "./restart_pysages.pkl"
restart_pysages_save_path: "./restart_pysages_2.pkl"
```

Set `restart_pysages_load_path` only together with `restart_load_path`.
Loading the `.npz` without the `.pkl` restarts the dynamics but discards
everything the method learned.

## Getting the free energy out

SO3LR does not compute free energies itself. Recover them from the `.pkl`
written to `restart_pysages_save_path` with PySAGES' own analysis:

```python
import pysages

result = pysages.serialization.load("restart_pysages.pkl")
fe = pysages.analyze(result)

# ABF and spectral ABF
grid = fe["mesh"]                 # CV values
free_energy = fe["free_energy"]   # in the simulation's energy units (eV)
```

For `metad`, `pysages.analyze` returns the deposited `heights` and a
`metapotential` function of the CV instead. The well-tempered free energy is
(T + ΔT)/ΔT × `metapotential(cv)`, up to sign and an additive constant.

For `harmonic` (umbrella sampling), SO3LR does not log the CV. Compute it from
the saved trajectory frames (one per `md_steps`) and reweight the windows
yourself with WHAM or MBAR.

ABF and metadynamics estimates drift for a long time before settling. Save
intermediate `.pkl` files and compare successive profiles: if the barrier is
still moving, the run is not converged. See the
[PySAGES documentation](https://pysages.readthedocs.io/) for the output keys of
each method.
