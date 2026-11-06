# Settings file reference

Every MD run can be driven by a single YAML file:

```shell
so3lr --settings settings.yaml
```

A settings file accepts every CLI flag, plus several features that have no
flag. We recommend it for production simulations.

An unknown key is ignored with a warning, not an error:

```
Unknown setting 'md_t' in settings.yaml; ignoring. Did you mean: md_dt, md_T, md_P?
```

It goes to stderr before the log file opens, so in a batch job look in the
scheduler's output file.

## A complete example

```yaml title="settings.yaml"
# --- what to run -------------------------------------------------------
ensemble: nvt                    # nvt | nvt-langevin | npt | nve
input_file: geometry.xyz
output_file: trajectory.hdf5
log_file: run.log

# --- model -------------------------------------------------------------
model: so3lr-tauri-m-2606              # pin the dated name for reproducibility
precision: float32
lr_cutoff: 12.0
dispersion_damping: 2.0
total_charge: 0

# --- dynamics ----------------------------------------------------------
md_T: 300                        # K
md_dt: 0.5                       # fs
md_cycles: 1000
md_steps: 1000
seed: 42

# --- thermostat --------------------------------------------------------
nhc_chain_length: 3
nhc_steps: 2
nhc_thermo: 100.0

# --- pre-run relaxation ------------------------------------------------
relax_before_run: true
force_convergence: 0.05          # eV/Å
min_cycles: 50                   # with min_steps: at most 500 FIRE steps

# --- checkpointing -----------------------------------------------------
restart_save_path: checkpoint.npz
save_buffer: 50
```

Only `input_file` is required.

## Key aliases

The settings file also accepts these CLI-style spellings:

| Alias | Internal key |
|---|---|
| `model` | `model_path` |
| `temperature` | `md_T` |
| `pressure` | `md_P` |
| `dt` | `md_dt` |
| `buffer_sr` | `buffer_size_multiplier_sr` |
| `buffer_lr` | `buffer_size_multiplier_lr` |
| `nhc_chain` | `nhc_chain_length` |
| `spin` | `num_unpaired_electrons` |
| `observable` | `observables` |
| `force_conv` | `force_convergence` |

If both an alias and its canonical key are given, the canonical key wins.

## Input and output

| Key | Default | Meaning |
|---|---|---|
| `input_file` | — | Geometry file in any format ASE can read. Required. |
| `output_file` | `<input>_<ensemble>.xyz` | Trajectory file, `.hdf5`/`.h5` or `.xyz`/`.extxyz`. See [trajectory files](md.md#trajectory-files). |
| `log_file` | the settings file's name with `.log` | For example, `npt.yaml` gives `npt.log`. |
| `overwrite` | `false` | Replace an existing output; a restart still appends to an existing EXTXYZ. |
| `output_atom_indices` | `null` | Write only these atoms to the trajectory (all atoms are still simulated). |
| `output_cartesian_positions` | `false` | Store Cartesian rather than fractional positions in periodic HDF5 output. |
| `ensemble` | inferred | `nvt` / `nvt-langevin` / `npt` / `nve`; if unset, `npt` when `md_P` is set and `nvt` otherwise. |

`output_format` is always set from the output extension.

## Model

| Key | Default | Meaning |
|---|---|---|
| `model_path` (`model`) | `so3lr-tauri-m-2606` | Bundled name or path. See [Models](../models.md). |
| `precision` | `float32` | `float32` (the model in float32, the MD state in float64) or `float64`. |
| `lr_cutoff` | `12.0` | Long-range cutoff in Å; use `100` for gas phase. |
| `dispersion_damping` | `2.0` | Dispersion switches off starting at `lr_cutoff - dispersion_damping` Å. |
| `total_charge` | `0` | Total charge of the system. |
| `num_unpaired_electrons` (`spin`) | `0` | Number of unpaired electrons (multiplicity − 1). |
| `charge_embed_total_charge` | `null` | Total charge seen only by the charge embedding (`null` follows `total_charge`). |
| `charge_equilibration_total_charge` | `null` | Total charge conserved by charge equilibration (`null` follows `total_charge`). |
| `buffer_size_multiplier_sr` | `1.25` | Short-range neighbour-list buffer. |
| `buffer_size_multiplier_lr` | `1.25` | Long-range neighbour-list buffer. |
| `minimum_cell_size_multiplier` | `1.0` | Cell-list cell size multiplier, applied to both lists. |
| `matmul_precision` | `highest` | JAX matmul precision, `default` / `high` / `highest`, set once per process. |

## Engine

| Key | Default | Meaning |
|---|---|---|
| `engine` | `auto` | `auto` / `flash` / `standard`. See [MD engines](md.md#md-engines). |
| `flash_mode` | `auto` | `auto` / `fast` / `default` / `lean` / `leanest` / `minimal`. |
| `flash_mixed` | `true` | In a float32 run, keep the MD state in float64. |
| `flash_devices` | `1` | Number of GPUs on one node to run flash on. |

The buffer and cell-list keys above apply to the neighbour-list engine only.

## Dynamics

| Key | Default | Meaning |
|---|---|---|
| `md_dt` (`dt`) | `0.5` | Timestep in fs. |
| `md_T` (`temperature`) | `300` | Target temperature in K. |
| `md_P` (`pressure`) | `1.0` under NPT | Target pressure in bar; setting it implies NPT unless another `ensemble` is set. |
| `init_T` | `md_T` | NVE only: temperature used to sample the initial velocities. |
| `md_cycles` | `100` | Number of cycles to run in this launch. |
| `md_steps` | `100` | Steps per cycle. |
| `save_buffer` | `50` | Frames buffered before a write; also the checkpoint interval. |
| `seed` | `52` | RNG seed. |
| `observables` (`observable`) | `[]` | Quantities to record. See [Observables](../advanced/observables.md). |

## Thermostat and barostat

| Key | Default | Meaning |
|---|---|---|
| `nhc_chain_length` (`nhc_chain`) | `3` | Nosé–Hoover chain length. |
| `nhc_steps` | `2` | Chain integration steps per MD step. |
| `nhc_thermo` | `100.0` | τ_thermostat = `md_dt × nhc_thermo`. |
| `nhc_baro` | `1000.0` | τ_barostat = `md_dt × nhc_baro`. |
| `nhc_sy_steps` | `3` | Suzuki–Yoshida steps. |
| `npt_coupling` | `isotropic` | `isotropic`, or the experimental `semi-isotropic` (see [flexible-cell NPT](md.md#flexible-cell-npt)). |
| `langevin_thermo` | `1.0` | Dimensionless timescale (> 0); the friction is γ = 1 / (`md_dt` × `langevin_thermo`). |
| `langevin_removecmmotion` | `true` | Remove centre-of-mass motion. |

## Pre-run relaxation

| Key | Default | Meaning |
|---|---|---|
| `relax_before_run` | `true` | Relax the geometry with FIRE before MD (skipped on a restart). |
| `force_convergence` (`force_conv`) | `null` | Stop once the max force falls below this, in eV/Å. |
| `min_cycles` | `10` | Maximum number of minimization cycles, a hard limit even with `force_convergence` set. |
| `min_steps` | `10` | Steps per minimization cycle. |
| `min_start_dt` | `0.05` | Initial FIRE timestep. |
| `min_max_dt` | `0.1` | Maximum FIRE timestep. |
| `min_n_min` | `2` | FIRE `n_min`. |

The relaxation frames are written to `<output-stem>_opt.xyz`.

## Restarts

| Key | Default | Meaning |
|---|---|---|
| `restart_save_path` | `null` | Where to write the `.npz` checkpoint. |
| `restart_load_path` | `null` | Existing checkpoint to resume from. |
| `restart_reset` | `false` | Keep positions, momenta, masses and box, but reset the thermostat/barostat state and cycle counter. |

See [what a restart preserves](md.md#what-a-restart-preserves).

## Atom constraints

| Key | Default | Meaning |
|---|---|---|
| `fixed_atom_indices` | `null` | 0-based indices of the atoms to freeze. |
| `mobile_atom_indices` | `null` | Alternatively, the atoms allowed to move (all others are frozen). |
| `atom_mask_path` | `null` | `.npz` file holding either array; explicit YAML keys override it. |
| `fixed_atom_constraint` | `cartesian_anchor` | How frozen atoms are pinned, `cartesian_anchor` or `fractional`. |

See [Freezing atoms](md.md#freezing-atoms) for an example.

## K-space electrostatics

| Key | Default | Meaning |
|---|---|---|
| `kspace_electrostatics` | `null` | `ewald` / `pme` / `p3m`. |
| `kspace_smearing` | `4.0` | Smearing width, Å. |
| `kspace_spacing` | `2.0` | Grid spacing, Å. |
| `kspace_interp_nodes` | `4` | Interpolation nodes for PME/P3M. |
| `kspace_npt_cycles` | `100` | Under NPT, rebuild the grid every N cycles (`null`: never); setting it moves the run off the flash engine. |

See [Long-range electrostatics](../advanced/electrostatics.md).

## K-space dispersion

The same methods can be applied to dispersion, for dense periodic systems.

| Key | Default | Meaning |
|---|---|---|
| `kspace_dispersion` | `null` | `ewald` or `pme`. |
| `kspace_dispersion_smearing` | `2.0` | Smearing width for the dispersion sum, Å. |
| `kspace_dispersion_spacing` | `3.0` | Grid spacing, Å. |
| `kspace_dispersion_interp_nodes` | `4` | Interpolation nodes. |
| `dispersion_tail_correction` | `false` | NPT only: analytic correction for the dispersion energy beyond the cutoff (not available for the legacy `so3lr`). |

## Stability monitoring

| Key | Default | Meaning |
|---|---|---|
| `energy_drift_threshold` | `0.1` | Warn when the drift rate exceeds this, in meV/atom/ps. |
| `drift_warmup_ps` | `5.0` | Equilibration time (ps) ignored before the reference energy is taken. |

## Enhanced sampling

| Key | Default | Meaning |
|---|---|---|
| `do_enhanced_sampling` | `false` | Turn on the PySAGES integration. |
| `pysages_input` | `./pysages_input.yaml` | Path to the PySAGES YAML that defines the CVs and method. |
| `restart_pysages_save_path` | `null` | Where to write the PySAGES state (`.pkl`). |
| `restart_pysages_load_path` | `null` | PySAGES state to resume from. |
| `enforce_cartesian_coordinates` | `false` | Forced on for periodic enhanced-sampling runs, except under NPT. |
| `cylinder_restraint_enabled` | `false` | Turn on the [cylindrical restraint](../advanced/enhanced-sampling.md#cylindrical-restraint). |
| `cylinder_restraint_atom_indices` | `null` | 0-based indices of the restrained atoms (their centre of mass); required. |
| `cylinder_restraint_radius` | `null` | Cylinder radius in Å; required. |
| `cylinder_restraint_force_constant` | `5.0` | Radial force constant, eV/Å². |
| `cylinder_restraint_axis` | `[0.0, 0.0, 1.0]` | Axis direction (any length). |
| `cylinder_restraint_axis_origin` | `null` | A point on the axis, Å; `null` is the box centre (required without a box). |
| `cylinder_restraint_axial_min`, `cylinder_restraint_axial_max` | `null` | Optional bounds along the axis, Å from the origin. |
| `cylinder_restraint_axial_force_constant` | `null` | Axial force constant, eV/Å²; `null` uses the radial one. |

See [Enhanced sampling](../advanced/enhanced-sampling.md).

## tune-ewald search ranges

`so3lr tune-ewald --settings tune.yaml` reads the model and k-space keys
above, plus these search ranges:

| Key | Default | Meaning |
|---|---|---|
| `accuracy` | `1e-3` | Target force error, eV/Å. |
| `cutoff_lo`, `cutoff_hi`, `cutoff_it` | `7.0` (`10.0` in the `cutoff` modes), `lr_cutoff`, `2.0` | Real-space cutoff range and step, Å. |
| `smear_lo`, `smear_hi`, `smear_it` | `2.0`, `6.0`, `2.0` | Smearing range and step, Å. |
| `ns_lo`, `ns_hi` | `4`, `8` | Ewald k-spacing = shortest cell vector / each value in this range. |
| `mesh_lo`, `mesh_hi` | `4`, `6` | PME/P3M mesh spacing = 2 × shortest cell vector / (2ⁿ − 1) for each *n* in this range. |
| `nodes_lo`, `nodes_hi` | `4`, `4` | Interpolation nodes (PME: 4 only, P3M: 3–5). |
| `dispersion_damping_lo`, `dispersion_damping_hi`, `dispersion_damping_it` | `2.0`, `4.0`, `1.0` | Damping range and step for the cutoff-only dispersion tuning, Å. |

## Evaluation keys

`so3lr eval` does not read `--settings`, but these keys are accepted without an
unknown-key warning:

| Key | Meaning |
|---|---|
| `datafile` | Dataset to evaluate. |
| `save_to` | Output extxyz file. |
| `targets` | Properties to score. |
| `batch_size` | Structures per batch. |
| `jit_compile` | JIT-compile the evaluation function. |
