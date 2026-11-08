# Molecular dynamics

SO3LR runs MD with [JAX-MD](https://github.com/jax-md/jax-md) integrators, and
the forces come from one of two [engines](#md-engines), both built for GPUs.
The trajectories can be saved in `.hdf5` or `.xyz` format. In addition,
checkpoints can be saved as `.npz` files throughout the simulation to restart
if needed.

## Ensembles

=== "NVT (Nosé–Hoover)"

    Run NVT (constant volume and temperature) simulation with the deterministic
    Nosé–Hoover chain thermostat. This is the default choice for equilibrium
    sampling.

    ```shell
    so3lr nvt --input geometry.xyz --temperature 300 \
        --dt 0.5 --md-cycles 100 --md-steps 100
    ```

=== "NVT (Langevin)"

    Run NVT simulation with the stochastic Langevin thermostat, which has
    stronger ergodicity.

    ```shell
    so3lr nvt-langevin --input geometry.xyz --temperature 300 \
        --dt 0.5 --md-cycles 100 --md-steps 100 --langevin-thermo 1.0
    ```

    Note that `--langevin-thermo` is a dimensionless timescale rather than the
    friction coefficient. The friction is γ = 1 / (`dt` × `langevin_thermo`).

=== "NPT"

    Run NPT (constant pressure and temperature) simulation with the
    Nosé–Hoover chain thermostat and barostat. It requires a periodic,
    orthorhombic cell.

    ```shell
    so3lr npt --input box.xyz --temperature 300 --pressure 1.0 \
        --dt 0.5 --md-cycles 100 --md-steps 100
    ```

=== "NVE"

    Run NVE (constant volume and energy) simulation, with no thermostat.

    ```shell
    so3lr nve --input geometry.xyz --temperature 300 \
        --dt 0.5 --md-cycles 100 --md-steps 100
    ```

    Here `--temperature` only sets the initial velocities.

A run is `md_cycles × md_steps` steps long, with one frame written per cycle.
Use many short cycles for frequent output, and a few long cycles for maximum
throughput.

Velocities or momenta in the input file (as in an EXTXYZ frame from an earlier
SO3LR trajectory) are used as the initial velocities. Otherwise, velocities
are drawn at the target temperature.

By default, the geometry is first relaxed with FIRE, capped at 10 × 10 steps
from the command line. Raise the cap in a
[settings file](settings.md#pre-run-relaxation), or skip the relaxation with
`--no-relax`.

### More NVT options

An NVT run with more of its options set:

```shell
so3lr nvt --input geometry.xyz --output trajectory.hdf5 --temperature 300 \
    --dt 0.5 --md-cycles 1000 --md-steps 1000 \
    --nhc-chain 3 --nhc-steps 2 --nhc-thermo 100.0 \
    --relax --force-conv 0.1 --seed 42 \
    --restart-save checkpoint.npz
```

### Flexible-cell NPT

Flexible-cell NPT is adapted from the jax-md extension of
[Bichelmaier, Carrete and Madsen, Phys. Rev. B 110, 174105 (2024)](https://doi.org/10.1103/PhysRevB.110.174105),
which follows Tuckerman et al. (2006) and Yu et al. (2010). By default, `npt` scales the box isotropically. Set the coupling with
`npt_coupling` in a settings file:

```yaml
ensemble: npt
npt_coupling: semi-isotropic   # isotropic (default) | semi-isotropic
md_P: 1.0
```

| `npt_coupling` | Cell degrees of freedom | Status |
|---|---|---|
| `isotropic` | One scale factor; cell shape fixed | Default |
| `semi-isotropic` | *x* and *y* scale together, *z* independently (membranes and slabs) | Experimental; validate before production use. |
| `anisotropic` | — | Rejected with an error. |

## MD engines

`so3lr nve`, `nvt`, `nvt-langevin` and `npt` (and `so3lr --settings`) run one of
two engines, which compute the same energy and forces:

- **flash**: fused GPU kernels written in JAX Pallas. On one A100 (40 GB) with
  `so3lr-tauri-m`, it is 3.7–4.0× faster than the neighbour-list engine and
  runs systems of up to 2.7 million atoms (9.0 million with the lowest-memory
  mode, `--flash-mode minimal`, and 84 million on one AMD MI355X with 288 GB).
- **standard**: the jax-md neighbour-list potential, which runs every setting on
  any JAX backend.

With `--engine auto` (the default), SO3LR runs flash whenever it supports the
other settings, and the neighbour-list engine otherwise. The log names the
engine and the reason:

```text
Engine: flash (auto)
Engine: neighbour-list potential (flash cannot: its NPT is isotropic, not semi-isotropic)
Engine: neighbour-list potential (--engine standard)
```

`--engine flash` or `--engine standard` forces one of the two. Forcing flash on
a run it cannot do raises an error instead of falling back. Under `auto`,
SO3LR also falls back when flash fails to start, for example with too little
memory.

### When flash runs

Flash runs only when all of the following hold:

| | Requirement |
|---|---|
| Device | An NVIDIA GPU with compute capability 8.0 or newer, or an AMD MI355X |
| Model | `so3lr-tauri-s/m/l` (both releases) or the legacy `so3lr` |
| Box | No box (an isolated molecule), or a periodic orthorhombic box at least 30 Å across at `lr_cutoff` 12 Å (25 Å at 10 Å; 31 Å under NPT) |
| Ensemble | `nve`, `nvt`, `nvt-langevin`, or `npt` with isotropic coupling |
| Long range | Real space, or PME (`--kspace-electrostatics pme`, `kspace_dispersion: pme`) |

NPT and k-space need a box. Semi-isotropic NPT, the dispersion tail correction,
`kspace_npt_cycles`, Ewald and P3M, fixed atoms and observables run on the
neighbour-list engine, and so does enhanced sampling outside the cases listed
in [Enhanced sampling](../advanced/enhanced-sampling.md#engines).

### Flash options

| Flag | Settings key | Default | Meaning |
|---|---|---|---|
| `--engine` | `engine` | `auto` | `auto`, `flash` or `standard`. |
| `--flash-mode` | `flash_mode` | `auto` | Memory/recomputation trade-off: `fast`, `default`, `lean`, `leanest` or `minimal`; `auto` picks the fastest that fits. |
| `--flash-devices` | `flash_devices` | `1` | Run on this many GPUs of one node. |

`--flash-devices N` speeds up a system that fits on one GPU (3.2–4.0× on 4
A100s). `CUDA_VISIBLE_DEVICES` selects the GPUs. Several GPUs
always run the `fast` memory mode. SO3LR raises `XLA_PYTHON_CLIENT_MEM_FRACTION`
when flash may run (see [environment variables](cli.md#environment-variables)).

## Box size

Both engines need every box length to be at least twice `lr_cutoff` (24 Å at
the default of 12 Å), and flash needs more (see
[When flash runs](#when-flash-runs)). A shorter box is refused. For a smaller box, you can:

- lower `--lr-cutoff` and use `--kspace-electrostatics pme` (plus
  `kspace_dispersion: pme` in a settings file). A cutoff below 10 Å needs
  k-space electrostatics.
- or use the [ASE calculator](ase.md#small-and-skewed-cells).

The cell must be orthorhombic. Mixed periodicity, such as
`pbc=[True, True, False]`, is an error. A structure with a cell but
`pbc=False` runs in vacuum.

## Trajectory files

The extension of `--output` picks the format: `.hdf5` / `.h5` for HDF5, and
`.xyz` / `.extxyz` for EXTXYZ. One frame is stored per cycle, and
`save_buffer` frames are buffered between writes. We recommend HDF5 for long
runs.

### HDF5 layout

HDF5 trajectories are frame-major:

| Dataset | Shape |
|---|---|
| `positions`, `momenta` | `(n_frames, n_atoms, 3)` |
| `box` | `(n_frames, 3)` for an isotropic orthorhombic cell, `(n_frames, 3, 3)` for a semi-isotropic one |
| one per recorded [observable](#recording-observables) | `(n_frames, n_atoms)`, except `x` is `(n_frames, n_atoms, features)` and `dipole_vec` `(n_frames, 1, 3)` |

Root attributes:

| Attribute | Content |
|---|---|
| `fractional_coordinates` | Whether `positions` are fractional (true for periodic runs unless `output_cartesian_positions: true`, or with enhanced sampling outside NPT) |
| `trajectory_layout` | `frame-major-v1` |
| `box_convention` | `ase-cell-vectors-as-rows` |
| `atomic_numbers`, `cell_a0` | Species of the written atoms, and the initial cell in Å |
| `ensemble`, `md_dt_fs`, `md_steps` | Ensemble, timestep in fs, MD steps per frame |
| `so3lr_version`, `input_file`, `model_path`, `observables` | Provenance |

Periodic runs store fractional positions by default. Convert them to Å:

```python
import h5py
import numpy as np

with h5py.File("trajectory.hdf5") as f:
    pos = f["positions"][:]             # (n_frames, n_atoms, 3)
    box = f["box"][:]
    if f.attrs["fractional_coordinates"]:
        cell = box if box.ndim == 3 else box[:, None, :] * np.eye(3)
        pos = pos @ cell                # rows of cell are lattice vectors; now Å
```

EXTXYZ frames are always in Cartesian Å.

### Relaxation output and overwriting

- The pre-run relaxation frames go to a separate `<output-stem>_opt.xyz`.
- An existing output is never replaced without `--overwrite`. With it, an HDF5
  file is replaced before integration starts, so a failed run loses the old
  one; use a new output path to keep it.

## Restarts

MD simulations can be restarted from a previously saved checkpoint. This is
useful for extending simulations or recovering from interruptions. To enable
restart:

```shell
# 1. First save a checkpoint during your simulation
#    (rewritten every `save_buffer` cycles and at the end)
so3lr nvt --input geometry.xyz --output traj.xyz --temperature 300 \
    --md-cycles 100 --restart-save checkpoint.npz

# 2. Then restart from this checkpoint to continue the simulation
so3lr nvt --input geometry.xyz --output traj.xyz --temperature 300 \
    --md-cycles 100 --restart-load checkpoint.npz --restart-save checkpoint_new.npz
```

The run resumes at the saved cycle and runs `--md-cycles` more, without the
pre-run relaxation. An EXTXYZ trajectory is appended to. An HDF5 trajectory
needs a new `--output` path, or `--overwrite`.

### What a restart preserves

The restart will continue the simulation from the exact state where it was
saved, preserving:

- positions, momenta, forces, masses and the box;
- the cycle/step counter;
- the thermostat and barostat state (NVT/NPT);
- the PRNG key for `nvt-langevin`;
- enhanced-sampling state, when `restart_pysages_save_path` is also set (a
  separate `.pkl`; keep it together with the matching checkpoint).

Loading a checkpoint into a different ensemble, or with `restart_reset: true`,
keeps positions, momenta, masses and the box, but starts the
thermostat/barostat state and the cycle counter fresh. An NPT checkpoint
resumes only with the `npt_coupling` it was written with.

## Freezing atoms

Any subset of atoms can be held fixed during `nve` / `nvt` / `npt`, for example
to keep a surface or a substrate rigid. Constraints can only be set in a
settings file:

```yaml title="settings.yaml"
ensemble: nvt
input_file: geometry.xyz
model: so3lr-tauri-m
md_T: 300
md_cycles: 100
md_steps: 100

# --- atom constraints (all optional; omit to run unconstrained) ---
fixed_atom_indices: [0, 1, 2]            # 0-based indices to freeze
# mobile_atom_indices: [3, 4, 5]         # or list the atoms allowed to move
# atom_mask_path: masks.npz              # or load either array from a .npz
fixed_atom_constraint: cartesian_anchor  # default
output_cartesian_positions: false
```

```shell
so3lr --settings settings.yaml
```

Give either `fixed_atom_indices` or `mobile_atom_indices`, and the complement
is derived. `cartesian_anchor` pins frozen atoms in real space, which stays
correct under NPT; the alternative is `fractional`.

## Writing only part of the system

`output_atom_indices` (0-based) restricts the saved trajectory to a subset of
atoms, such as a solute in a large solvent box. MD still propagates the full
system.

```yaml title="settings.yaml — save only the solute"
output_atom_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Recording observables

Any [observable](../advanced/observables.md) can be recorded alongside the
trajectory (or with `observables:` in a settings file):

```shell
so3lr nvt --input geometry.xyz --temperature 300 \
    --observable nn_energy --observable partial_charges
```

## Monitoring stability

If an NVE or NVT run heats or cools steadily, it usually means that the
timestep is too large or the neighbour-list buffers are too small. SO3LR
warns once when the drift of the conserved quantity exceeds a threshold:

```yaml
energy_drift_threshold: 0.1     # meV/atom/ps (default); warn above this rate
drift_warmup_ps: 5.0            # ignore the first 5 ps while equilibrating
```

If the warning triggers, try halving `dt`, raising `--buffer-sr` / `--buffer-lr`,
or switching to `--precision float64`. A run whose energy or temperature
becomes NaN stops with an error.

## Choosing a timestep

A timestep of `0.5` fs is a safe default for systems containing hydrogen.
SO3LR has no constraint algorithm such as SHAKE or RATTLE, so do not use 2 fs
as you would with a classical force field.

## Related

- [Settings file reference](settings.md): every key
- [Command line](cli.md#running-on-a-cluster): batch jobs and chaining restarts
- [Long-range electrostatics](../advanced/electrostatics.md): Ewald/PME/P3M for periodic systems
- [Enhanced sampling](../advanced/enhanced-sampling.md): biased MD and free energies
- [Examples](../examples.md): notebooks for NVT and NPT production runs
