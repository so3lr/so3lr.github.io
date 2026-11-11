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

Enhanced-sampling code runs only when `do_enhanced_sampling: true` is set. See
[Engines](#engines) for which MD engine runs it.

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
| `gyrrad` | `indices` | Radius of gyration; cannot be biased (see [Limitations](#limitations)) |
| `princmom` | `indices`, `axis` | Principal moment of inertia |
| `component` | `indices`, `axis` | Cartesian component (`axis` 0/1/2 = x/y/z) of a group's barycentre |
| `relativecomponent` | `grp1`, `grp2`, `axis` | Same, relative to a second group |
| `asphericity` | `indices` | Asphericity |
| `acylindricity` | `indices` | Acylindricity |
| `shapeanisotropy` | `indices` | Shape anisotropy |
| `coordinationnumber` | `indices` | Coordination number, with element filtering (`species`, `cn_exps`, `indices_nn`, `r0_multiplier`; see [Coordination number](#coordination-number)) |
| `ringphase`, `ringamplitude` | `indices` | Ring pucker phase angle and amplitude |
| `chargeseparation` | `reference` | Signed distance between an acid or base and the H₃O⁺ or OH⁻ it formed in the water (see [Proton-transfer CVs](#proton-transfer-cvs)) |
| `ionsitedistance` | `indices` | Distance between a site and the H₃O⁺ or OH⁻ in the water |

CV type names are case-insensitive. The minimum-image variants `distance_pbc`,
`differenceofdistances_pbc`, `meanofdistances_pbc` and `relativecomponent_pbc`
take the same keys and require a periodic box.

Atom indices are **0-based** in the input geometry's order, given as plain
integer lists. Every CV except the two-group ones above and `chargeseparation`
takes a single flat `indices` list, so a `dihedral` is `indices: [4, 6, 8, 14]`.
The `grp1` and `grp2` of `relativecomponent` and `relativecomponent_pbc`, and
the atom keys of `chargeseparation` and `ionsitedistance`, also accept
inclusive ranges, such as `grp2: "0-19"` or `grp1: [0, "3-5"]`.

### Coordination number

`coordinationnumber` sums (1 − (r/r₀)ⁿ) / (1 − (r/r₀)ᵐ) over the neighbours of
the `indices` atoms, where r₀ is `r0_multiplier` times the sum of the two
atoms' covalent radii.

| Key | Default | Meaning |
|---|---|---|
| `species` | all elements | Count only neighbours of this element (atomic number). |
| `indices_nn` | all atoms | Count only these atoms as neighbours. |
| `cn_exps` | `[6, 12]` | The exponents n and m; m must be a multiple of n. |
| `r0_multiplier` | `1.6` | Factor on the summed covalent radii. |

Only neighbours within the model's short-range cutoff count: 4.5 Å for
`so3lr-tauri-s/m` and the legacy `so3lr`, 5.0 Å for `so3lr-tauri-l`. Keep r₀
well inside it, or the CV jumps where a neighbour crosses the cutoff.

The CV builds its neighbour list again at every step, over the whole system
and without a cell list, so its cost and memory grow with the square of the
number of atoms. That is cheap for a few thousand atoms; for tens of thousands
it can dominate the step.

### Proton-transfer CVs

`chargeseparation` and `ionsitedistance` follow the proton that an acid gives
to the surrounding water (or the proton a base takes from it). Both share every
H atom among the heavy atoms with Gaussian weights of width `sigma`, so that a
water oxygen owns about 2 H, an H₃O⁺ 3 and an OH⁻ 1. Both need a periodic
orthorhombic box at constant volume: they keep the box of the start of the run,
so they are refused under NPT. The water oxygens are `water_indices` if given,
otherwise every O atom with no other heavy atom within 1.7 Å at the start of
the run; with none, both CVs stop with an error. The log lists the atoms each
CV was built from.

`chargeseparation` ([Xu et al., arXiv:2610.13265](https://arxiv.org/abs/2610.13265))
is s = Σᵢ qᵢ dᵢ, where qᵢ is the H count of charge site i minus its neutral
value and dᵢ its distance from the centre of the `reference` atoms. s ≈ 0 for
the undissociated acid or base, s ≈ +d with an H₃O⁺ at distance d, and s ≈ −d
with an OH⁻.

| Key | Default | Meaning |
|---|---|---|
| `reference` | — | Atoms of the acid or base centre (required). |
| `sigma` | `0.3` | Width of the H assignment, Å. |
| `owner_species` | `[6, 7, 8, 15, 16]` | Elements whose atoms own H (C, N, O, P, S). |
| `water_indices` | automatic | The water oxygens. |
| `charge_sites` | `water` | Whose charge enters s: `water` (the water oxygens), `all` (also every other owner atom except `reference`), or a list of atoms to add to the water oxygens. |
| `neutral_reference` | the input structure (after the relaxation before MD, if one runs) | With `charge_sites` other than `water`: an undissociated structure that gives each site's neutral H count. |
| `pair_guard` | `false` | Map configurations with both an H₃O⁺ and an OH⁻ in the water (autoionisation) onto s = 0. Not recommended with ABF. |
| `n_hydronium`, `n_hydroxide` | `2.1`, `1.9` | Pair guard: H counts above / below which a water starts to count as H₃O⁺ / OH⁻. |
| `guard_width`, `guard_x0` | `0.8`, `0.5` | Pair guard: width of that step, and where s reaches 0. |

With `charge_sites` other than `water`, put every atom that can carry the
titratable proton into `reference` (both carboxyl O, both His ring N);
otherwise its proton counts as a charge.

`ionsitedistance` is the distance between the site and the H₃O⁺
(`mode: acid`) or OH⁻ (`mode: base`) in the water, and the distance to the
nearest water while there is no such ion. It is meant to be biased together
with a `coordinationnumber` of the same site, which drives the proton transfer.

| Key | Default | Meaning |
|---|---|---|
| `indices` | — | Heavy atoms of the site, such as both carboxyl O (required). |
| `mode` | `acid` | `acid` follows the H₃O⁺, `base` the OH⁻. |
| `n0` | `2.5` (acid), `1.5` (base) | Water H count at which a water counts half as the ion. |
| `a` | `0.4` | Half-width of that step. Acid needs n0 − a ≥ 2 and n0 + a ≤ 3, base n0 + a ≤ 2 and n0 − a ≥ 1. |
| `sigma` | `0.25` | Width of the H assignment, Å. |
| `tau` | `0.2` | Softness of the nearest-water distance, Å. |
| `water_indices` | automatic | The water oxygens. |

## Supported ensembles

`abf`, `spectralabf` and `metad` require NVT (Nosé–Hoover, `ensemble: "nvt"`).
`harmonic` and `unbiased` also run under `npt` and `nve`, except with
`chargeseparation` or `ionsitedistance`, which need a constant volume.
`nvt-langevin` is not supported by any method.

## Engines

With `engine: auto` on a GPU, enhanced sampling runs on the
[flash engine](../usage/md.md#md-engines) when, in addition to the
[usual conditions](../usage/md.md#when-flash-runs):

- the ensemble is `nvt` or `nve`;
- the box is periodic;
- `flash_devices` is `1`, and `flash_mode` is `auto` or `fast` (with a system
  that fits the fast mode);
- no [cylindrical restraint](#cylindrical-restraint) is set.

Otherwise it runs on the neighbour-list engine, and `engine: flash` stops with
the reason. The two engines compute the same forces up to rounding, so their
trajectories drift apart over time, as runs on two different GPUs do. Set
`engine: standard` to reproduce a run made on the neighbour-list engine.
Checkpoints and `.pkl` files continue on either engine.

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

## Cylindrical restraint

For host–guest binding, a flat-bottom cylinder keeps the guest near the axis
of the host while a method samples along it. It acts on the mass-weighted
centre of mass of the listed atoms: zero inside a cylinder of radius R (and
between the optional axial bounds), harmonic outside. At a distance r > R from
the axis the energy is ½ k_r (r − R)², plus ½ k_z times the squared distance
beyond an axial bound. Set it in the MD settings file:

```yaml title="md_settings.yaml"
cylinder_restraint_enabled: true
cylinder_restraint_atom_indices: [120, 121, 122]   # the guest, 0-based
cylinder_restraint_radius: 5.0                     # R in Å (required)
cylinder_restraint_force_constant: 5.0             # k_r in eV/Å² (default 5.0)
cylinder_restraint_axis: [0.0, 0.0, 1.0]           # default: z
# cylinder_restraint_axis_origin: [17.5, 17.5, 17.5]  # Å; default: the box centre
# cylinder_restraint_axial_min: -10.0                 # Å along the axis from the origin
# cylinder_restraint_axial_max: 10.0
# cylinder_restraint_axial_force_constant: 5.0        # k_z; default: k_r
```

The axis and origin are fixed in space, so freeze the host with
[fixed atoms](../usage/md.md#freezing-atoms) to keep it on the axis. The atoms
are a list, a single index or a range string such as `"120-135"` (inclusive);
an atom listed twice is an error. In a periodic box, the guest is made whole
around its first atom before its centre of mass is taken, so it must span less
than half the box, and its distance from the origin uses the minimum image;
without a box, set the origin. The restraint energy is part of the potential
energy in the log. The restraint is not applied during the relaxation before
MD, and it runs on the neighbour-list engine. The keys are listed in the
[settings reference](../usage/settings.md#enhanced-sampling).

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

A PySAGES restart must stay in the same ensemble, and it continues with the
method stored in the `.pkl` (CVs, grid, restraints and method arguments).
Changing `pysages_input` between the runs has no effect; SO3LR logs a warning
for every setting that differs.

## The CV log

Every run appends the CV values of each MD step to `cv_vs_timestep.txt` in the
working directory: one row per step, one column per CV in the order of
`pysages_input`. The file is never overwritten, so a continuation adds to it,
and runs started in the same directory write into the same file; give each
replica or window its own directory. A cycle that is repeated (the log reports
a neighbour-list overflow or, on flash, "cycle repeated") adds no rows for the
discarded attempt: they go to `cv_vs_timestep_discarded.txt`, under a comment
line giving the steps, the attempt and the reason, so `cv_vs_timestep.txt`
keeps one row per step.

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

For `harmonic` (umbrella sampling), take the CV values from the
[CV log](#the-cv-log) and reweight the windows yourself with WHAM or MBAR.

ABF and metadynamics estimates drift for a long time before settling. Save
intermediate `.pkl` files and compare successive profiles: if the barrier is
still moving, the run is not converged. See the
[PySAGES documentation](https://pysages.readthedocs.io/) for the output keys of
each method.

## Limitations

These two come from PySAGES itself, not from SO3LR's interface to it:

- `gyrrad` cannot be biased: PySAGES' radius of gyration returns three values,
  not one.
- `spectralabf` with two CVs fails in PySAGES with JAX 0.10, the version in
  SO3LR's lock file; with one CV it works.

The CVs that take the box keep the box of the start of the run. Under NPT,
which `harmonic` and `unbiased` allow, the `_pbc` variants and
`coordinationnumber` then measure distances between atoms on opposite sides of
the box edge off by the change in box length, so use them at constant volume.
`chargeseparation` and `ionsitedistance` are refused under NPT.
