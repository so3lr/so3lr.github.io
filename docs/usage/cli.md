# Command line

SO3LR provides one command-line interface for geometry optimization,
molecular dynamics, evaluation, training and parameter tuning.

## Commands

| Command | What it does |
|---|---|
| `so3lr opt` | Geometry optimization (FIRE) |
| `so3lr nvt` | NVT molecular dynamics with the Nosé–Hoover chain thermostat |
| `so3lr nvt-langevin` | NVT molecular dynamics with the Langevin thermostat |
| `so3lr npt` | NPT molecular dynamics with the Nosé–Hoover chain thermostat and barostat |
| `so3lr nve` | NVE molecular dynamics |
| `so3lr eval` | Model evaluation on a dataset |
| `so3lr finetune` | Fine-tune a model on custom data |
| `so3lr train` | Train a SO3LR/SO3krates model from scratch |
| `so3lr tune-ewald` | Auto-tune the Ewald/PME/P3M k-space parameters |

```shell
so3lr --help          # command overview
so3lr --help-full     # every command with full example commands
so3lr nvt --help      # all flags for one command (-h works too)
```

## Two ways to drive a run

Use flags for short, exploratory runs:

```shell
so3lr nvt --input geometry.xyz --temperature 300 \
    --dt 0.5 --md-cycles 100 --md-steps 100
```

For production runs, use a settings file. It accepts every flag, plus
features that have no flag:

```shell
so3lr --settings settings.yaml
```

The keys are listed in the [Settings file reference](settings.md), and
[Molecular dynamics](md.md) has examples of both.

## Shared options

These options are accepted by `opt`, the four MD commands and `tune-ewald`.
`eval` takes a subset (see [Evaluation](#evaluation-eval)).

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model`, `--model-path` | `so3lr-tauri-m-2606` | Bundled name or path to a custom model. See [Models](../models.md). |
| `--precision` | `mixed` | `float32` or `float64` throughout, or `mixed`: a float32 model around a float64 MD state (positions, momenta, box, energy sums). |
| `--lr-cutoff` | `12.0` | Long-range cutoff in Å. |
| `--dispersion-damping` | `2.0` | Dispersion switches off starting at `lr_cutoff - dispersion_damping` Å. |
| `--buffer-sr` | `1.25` | Short-range neighbour-list buffer multiplier. |
| `--buffer-lr` | `1.25` | Long-range neighbour-list buffer multiplier. |
| `--total-charge` | `0` | Total charge of the system (integer). |
| `--spin` | `0` | Number of unpaired electrons (multiplicity − 1). |
| `--charge-embed-total-charge` | follows `--total-charge` | Total charge seen only by the charge embedding. |
| `--charge-equilibration-total-charge` | follows `--total-charge` | Total charge conserved by charge equilibration. |

The default 12 Å cutoff is meant for periodic systems. For gas phase, use
`--lr-cutoff 100`, otherwise electrostatics and dispersion are truncated. A
periodic box must be at least 2 × `lr_cutoff` across (see
[Box size](md.md#box-size)).

### Input and output

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `--input_file` | — | Geometry file in any format that [`ase.io.read`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) can read. |
| `--output`, `--output_file` | `<input>_<ensemble>.xyz` | Trajectory file, `.hdf5` or `.xyz` (suffix `_nvt_langevin` or `_opt` for those commands). |
| `--log-file` | `<output>` with a `.log` extension | Log file; with `--settings`, it is named after the YAML file. |
| `--overwrite` / `--no-overwrite` | `--no-overwrite` | Replace an existing output; a restart appends to an existing EXTXYZ (see [trajectory files](md.md#relaxation-output-and-overwriting)). |

### K-space electrostatics

`--kspace-electrostatics` (`ewald` / `pme` / `p3m`, off when unset),
`--kspace-smearing`, `--kspace-spacing`, `--kspace-interp-nodes` and
`--kspace-npt-cycles`. `eval` takes the first four, plus `--kspace-dispersion`
and its `-smearing`, `-spacing` and `-interp-nodes` flags; for MD, k-space
dispersion is set in the settings file. Defaults and tuning are in
[Long-range electrostatics](../advanced/electrostatics.md).

## MD options

These options are shared by `nvt`, `nvt-langevin`, `npt` and `nve`.

| Flag | Default | Meaning |
|---|---|---|
| `--temperature` | `300` | Temperature in K. Under `nve`, the temperature used to initialize velocities. |
| `--dt` | `0.5` | Timestep in fs. |
| `--md-cycles` | `100` | Number of cycles to run in this launch. |
| `--md-steps` | `100` | Steps per cycle (total steps = cycles × steps). |
| `--save-buffer` | `50` | Frames (cycles) buffered before writing; also the checkpoint interval. |
| `--seed` | `52` | RNG seed. |
| `--relax` / `--no-relax` | `--relax` | Relax the geometry before starting MD (skipped on a restart). |
| `--force-conv` | `None` | Force convergence criterion (eV/Å) for the initial relaxation. |
| `--restart-save` | `None` | Write a `.npz` checkpoint. |
| `--restart-load` | `None` | Resume from an existing `.npz` checkpoint. |
| `--observable` | none | Extra quantity to record at each save step, repeatable (see [Observables](../advanced/observables.md)). |

Atom constraints, trajectory subsetting, k-space dispersion, flexible-cell
NPT, the relaxation's FIRE budget and enhanced sampling have no MD flag; set
them in a [settings file](settings.md).

### Engine

See [MD engines](md.md#md-engines).

| Flag | Default | Meaning |
|---|---|---|
| `--engine` | `auto` | `auto`, `flash` or `standard`. |
| `--flash-mode` | `auto` | `fast`, `default`, `lean`, `leanest` or `minimal` (fastest first); `auto` picks the fastest that fits in memory. |
| `--flash-devices` | `1` | Number of GPUs on one node to run flash on. |
| `--flash-kernels` | `auto` | `auto`, `pallas` or `cuda`: flash's [CUDA kernels](md.md#cuda-kernels) on an H100/H200 where they apply. |

### Thermostat — `nvt` and `npt`

| Flag | Default | Meaning |
|---|---|---|
| `--nhc-chain` | `3` | Nosé–Hoover chain length. |
| `--nhc-steps` | `2` | Chain integration steps per MD step. |
| `--nhc-thermo` | `100.0` | Thermostat damping multiplier, τ = `dt × nhc_thermo`. |
| `--nhc-sy-steps` | `3` | Suzuki–Yoshida integration steps. |

### Barostat — `npt`

| Flag | Default | Meaning |
|---|---|---|
| `--pressure` | `1.0` | Pressure in bar. |
| `--nhc-baro` | `1000.0` | Barostat damping multiplier, τ = `dt × nhc_baro`. |

From the command line, the box couples isotropically (semi-isotropic is a
[settings-file key](md.md#flexible-cell-npt)).

### Langevin thermostat — `nvt-langevin`

| Flag | Default | Meaning |
|---|---|---|
| `--langevin-thermo` | `1.0` | Dimensionless timescale (> 0, not γ); the friction is γ = 1 / (`dt` × `langevin_thermo`). |
| `--langevin-removecmmotion` | `True` | Remove centre-of-mass motion. |

[Enhanced sampling](../advanced/enhanced-sampling.md) is not available with
`nvt-langevin`.

## Geometry optimization — `opt`

Optimize a structure with FIRE. Besides the shared options, `opt` takes:

| Flag | Default | Meaning |
|---|---|---|
| `--force-conv` | `None` | Stop once the max force falls below this, in eV/Å. |
| `--min-cycles` | `10` | Maximum number of minimization cycles. |
| `--min-steps` | `10` | Steps per cycle. |
| `--min-start-dt` | `0.05` | Initial FIRE timestep. |
| `--min-max-dt` | `0.1` | Maximum FIRE timestep. |
| `--min-n-min` | `2` | FIRE `n_min` parameter. |

```shell
so3lr opt --input geometry.xyz --force-conv 0.05 --lr-cutoff 12.0
```

`--min-cycles` × `--min-steps` is a hard limit: if it is reached before
`--force-conv`, a warning is logged and the last positions are kept. The output
is an EXTXYZ trajectory with one frame per cycle. See the
[JAX-MD FIRE docs](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html#jax_md.minimize.fire_descent).

## Evaluation — `eval`

Evaluate a model on a dataset:

```shell
so3lr eval --datafile dataset.extxyz --batch-size 1 \
    --lr-cutoff 100 --save-to predictions.extxyz
```

The main flags are below; see [Dataset evaluation](evaluation.md) for all of them.

| Flag | Default | Meaning |
|---|---|---|
| `--datafile` | — | Any file that `ase.io.iread` can read. |
| `--save-to` | `None` | Output `.extxyz` with predictions. |
| `--targets` | `forces,dipole_vec,hirshfeld_ratios` | Comma-separated properties to score and save. |
| `--batch-size` | `1` | Structures per batch. |
| `--jit-compile` / `--no-jit-compile` | on | Disable to debug shape errors. |
| `--total-charge`, `--spin` | from file | Override the per-structure values. |
| `--overwrite` / `--no-overwrite` | refuse | Replace existing outputs. |
| `--log-file` | next to `--save-to` | If nothing is saved, the log goes to the console only. |

`eval` also takes `--model`, `--precision`, `--lr-cutoff`,
`--dispersion-damping`, the two charge overrides and the k-space flags.

## Training and fine-tuning

See [Fine-tuning](../training/finetuning.md) and
[Training from scratch](../training/from-scratch.md).

```shell
so3lr finetune --datafile dataset.xyz --workdir so3lr_finetuned \
    --num-train 1000 --num-valid 100
so3lr train --config config.yaml
```

| Command | Flags |
|---|---|
| `finetune` | `--datafile`, `--workdir`, `--num-train`, `--num-valid`, `--strategy` (default `full`), `--config` (YAML; default the packaged `finetune.yaml`), `--model`, `--log-file` (default `<workdir>/finetuning.log`) |
| `train` | `--config` (YAML or JSON), `--log-file` (default: the config name with `.log`) |

## Tuning k-space parameters

See [Long-range electrostatics](../advanced/electrostatics.md#automatic-tuning).

```shell
so3lr tune-ewald --input water_box.xyz --kspace-electrostatics pme \
    --lr-cutoff 12.0 --precision float64
```

`tune-ewald` also accepts `--jit-compile` / `--no-jit-compile` and
`--settings`, which reads the shared keys and the
[search ranges](settings.md#tune-ewald-search-ranges) from YAML. Its log
defaults to `<input>_tune-kspace.log`.

## The log file

Every MD run writes a log file next to its output. The header records the
settings of the run:

```text
============================================================
SO3LR NVT Molecular Dynamics Simulation (v0.2.0)
============================================================
Initial geometry:          geometry.xyz
Output file:               geometry_nvt.xyz
Log file:                  geometry_nvt.log
Force field:               so3lr-tauri-m-2606 (bundled)
Precision:                 mixed
Long-range cutoff:         12.0 Å
Dispersion damping:        2.0 Å
Short-range buffer:        1.25
Long-range buffer:         1.25
Total charge:              0
Spin (unpaired e-):        0
Temperature:               300.0 K
Ensemble:                  NVT
Simulation length:         100000 steps (50.00 ps)
MD cycles:                 1000
Steps per cycle:           100
Timestep:                  0.5 fs
...
```

It continues with the full merged settings (which can be pasted back into a
settings file), the JAX backend and devices, and the [engine](md.md#md-engines)
used. The model is logged under the name you gave it, so pass a dated name such
as `so3lr-tauri-m-2606` to identify the release.

Please attach the header when reporting a problem.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Completed successfully |
| `1` | The run failed; the error is printed to stderr. |
| `2` | Usage error (unknown flag or model, bad choice, missing input); nothing runs. |

So you can chain commands in the shell or with scheduler dependencies:

```shell
so3lr opt --input geometry.xyz --output min.xyz \
  && so3lr nvt --input min.xyz --output nvt.xyz --md-cycles 1000
```

## Running on a cluster

One simulation runs on one GPU, except that the flash engine can spread it
over several GPUs of a node with `--flash-devices` (see
[MD engines](md.md#flash-options)).

```bash title="so3lr.sbatch"
#!/bin/bash
#SBATCH --job-name=so3lr-nvt
#SBATCH --gpus=1
#SBATCH --time=24:00:00

module load CUDA                 # site-specific
source /path/to/venv/bin/activate

srun so3lr --settings settings.yaml
```

### Surviving the walltime limit

Split a long run into a chain of jobs, each resuming from the previous
checkpoint. The script checks that the checkpoint exists first:

```bash title="chain.sbatch"
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=24:00:00

resume=()
[ -f checkpoint.npz ] && resume=(--restart-load checkpoint.npz)

srun so3lr nvt --input geometry.xyz --output traj.xyz \
    --md-cycles 1000 --md-steps 1000 --save-buffer 50 \
    --restart-save checkpoint.npz "${resume[@]}"
```

```shell
jid=$(sbatch --parsable chain.sbatch)
for i in $(seq 1 9); do
    jid=$(sbatch --parsable --dependency=afterok:$jid chain.sbatch)
done
```

Each job runs `--md-cycles` more cycles and appends to `traj.xyz` (an `.hdf5`
output needs a new path per job). Set `--save-buffer` so checkpoints are written
well within the walltime: a killed job loses everything since the last one. See
[restarts](md.md#restarts).

### Environment variables

These JAX/XLA variables are useful on shared nodes.

| Variable | Use |
|---|---|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of GPU memory JAX may take (default `0.75`, raised by SO3LR to `0.97`, or `0.90` with `--flash-devices`, when flash may run and it is unset); lower it to share a GPU. |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Allocate memory on demand, which suits a shared GPU but is slower. |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to the process. |
| `JAX_PLATFORMS=cpu` | Force CPU even when a GPU is available. |
| `SO3LR_CUDA_CACHE` | Where flash's [CUDA kernels](md.md#cuda-kernels) are built (default `~/.cache/so3lr/flash_cuda`). |
| `SO3LR_NVCC`, `CUDA_HOME` | Locate `nvcc` for the CUDA kernels when it is not on the `PATH`. |

If a run is unexpectedly slow, check the backend and device at the start of
the log.

## Flag spellings

Flags are `kebab-case` (`--md_cycles` is rejected). These synonyms are
accepted:

| Canonical | Also accepted |
|---|---|
| `--input` | `--input_file` |
| `--output` | `--output_file` |
| `--model` | `--model_path`, and also `--model-path` except on `eval` and `finetune` (`train` has no `--model`) |
| `--datafile` | `--data-file`, `--datafile_path` |
| `--workdir` | `--work-dir`, `--workdir_path` |
| `--config` | `--config_path` (`finetune` only) |
| `--min-cycles`, `--min-steps` | `--min_cycles`, `--min_steps` |
| `--min-start-dt` | `--dt-start` |
| `--min-max-dt` | `--dt-max` |
| `--num-train`, `--num-valid` | `--num_train`, `--num_valid` |
| `--help` | `-h` |

Settings-file keys are always `snake_case`.
