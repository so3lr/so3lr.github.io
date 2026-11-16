# Command line

SO3LR provides one command-line interface for geometry optimization,
molecular dynamics, evaluation, training and parameter tuning.

## Commands

| Command | What it does |
|---|---|
| `so3lr start` | Set up MD, an optimization or an evaluation through menus, then start it or save it as a run folder |
| `so3lr show PATH` | Follow MD, optimization and training runs from their journals, live or afterwards (`--json` for scripts) |
| `so3lr resume PATH` | Continue a settings-file MD run from its last checkpoint |
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
so3lr --help-full     # command overview with additional usage examples
so3lr nvt --help      # flags grouped by purpose (-h works too)
so3lr --version       # installed SO3LR version
```

## Two ways to drive a run

Use flags for short, exploratory runs:

```shell
so3lr nvt --input geometry.xyz --temperature 300 \
    --dt 0.5 --md-cycles 100 --md-steps 100
```

For production runs, use a settings file. It accepts simulation settings,
including features that have no flag:

```shell
so3lr --settings settings.yaml
```

The keys are listed in the [Settings file reference](settings.md), and
[Molecular dynamics](md.md) has examples of both.

Use `--settings` without a subcommand. Flags and settings keys share their
defaults, so leaving out a key gives the same run as leaving out its flag. This
includes the [long-range defaults](#long-range-defaults), which `opt`, `eval`
and `so3lr start` use too.

## Guided setup and monitoring

`so3lr start` sets up MD, an optimization or an evaluation through menus and
saves it as a folder with a `run.sh`; `so3lr start --input STRUCTURE` does the
same without prompts. In a terminal, runs show live progress and graphs, and
`so3lr show` follows MD, optimization and training runs from their journals.
The display options `--display`, `--verbose`/`-v`, `--quiet`/`-q` and
`--no-color` may go before or after the subcommand. All of this is described in
[Guided setup and monitoring](terminal-ui.md).

## Resuming a run

`so3lr resume` continues a settings-file MD run from its checkpoint for the rest
of its `md_cycles`:

```shell
so3lr resume so3lr-run                 # a folder made by so3lr start
so3lr resume settings.yaml             # or any settings file
so3lr resume --or-start so3lr-run      # start it instead when it has no checkpoint yet
```

`PATH` is a folder with a `settings.yaml` (its paths are relative to the
folder), or a settings file (paths relative to the current directory, as with
`so3lr --settings`); the default is the current directory. The settings must set
`restart_save_path`; `resume` reads the cycle stored in that checkpoint, then:

- runs the remaining `md_cycles` from the checkpoint, without the relaxation;
- appends to an EXTXYZ trajectory, even when the file sets `overwrite: true`;
- writes an HDF5 trajectory to a new file, `<stem>.from_cycle_N.h5` (with
  `-2`, `-3`, ... when that exists);
- with enhanced sampling, continues the PySAGES bias from
  `restart_pysages_save_path`, and refuses to start when that file is missing;
- prints `Already complete` and exits when the checkpoint is at the end.

It refuses a settings file that already sets `restart_load_path`, since the
original length is then unknown; set `restart_load_path` and `md_cycles` by hand
there. Without a checkpoint yet, `resume` is a usage error, and
`resume --or-start` starts the run from the beginning, replacing the outputs of
any earlier attempt (a file without `restart_save_path` just runs). A run folder's `run.sh` runs `so3lr resume --or-start .`,
so the same script starts and continues the run. Runs started with flags resume
with `--restart-load` (see [Restarts](md.md#restarts)).

## Shared options

These options are accepted by `opt`, the four MD commands and `tune-ewald`.
`eval` takes a subset (see [Evaluation](#evaluation-eval)).

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model`, `--model-path` | `so3lr-tauri-m-2606` | Bundled name or path to a custom model. See [Models](../models.md). |
| `--precision` | `mixed` | `float32` or `float64` throughout, or `mixed`: a float32 model around a float64 MD state (positions, momenta, box, energy sums). |
| `--lr-cutoff` | `10.0` with PME, else `12.0` | Long-range (real-space) cutoff in Å; see [Long-range defaults](#long-range-defaults). |
| `--dispersion-damping` | `2.0` | Dispersion switches off starting at `lr_cutoff - dispersion_damping` Å. |
| `--buffer-sr` | `1.25` | Short-range neighbour-list buffer multiplier. |
| `--buffer-lr` | `1.25` | Long-range neighbour-list buffer multiplier. |
| `--total-charge` | `0` | Total charge of the system (integer). |
| `--spin` | `0` | Number of unpaired electrons (multiplicity − 1). |
| `--charge-embed-total-charge` | follows `--total-charge` | Total charge seen only by the charge embedding. |
| `--charge-equilibration-total-charge` | follows `--total-charge` | Total charge conserved by charge equilibration. |

MD and `opt` keep the charge and spin you pass, but warn when they contradict
the values recorded in the structure file (`charge`, `multiplicity` or `spin`,
read as in `so3lr eval`) or describe an impossible electron count.

For a molecule in vacuum, use `--lr-cutoff 100`; at 12 Å its electrostatics
and dispersion are truncated. On the neighbour-list engine, a periodic box must
be at least 2 × `lr_cutoff` across; flash takes smaller boxes (see
[Box size](md.md#box-size)).

### Long-range defaults

Unless you set them, the long-range settings follow the structure. The same rule
applies to the MD commands, `so3lr --settings`, `opt`, `eval` and
[`so3lr start`](terminal-ui.md#molecular-dynamics):

| Structure | Electrostatics and dispersion | Cutoff | Smearing and mesh |
|---|---|---|---|
| Periodic along all three axes | PME for both | 10 Å; in a cell too small for that, 0.45 × the smallest cell height (rounded down to 0.1 Å, at least the model's short-range cutoff) | Electrostatics 2 Å and 1.55 Å (1 Å below a 10 Å cutoff); dispersion 2 Å and 3 Å |
| Not periodic | real space | 12 Å; use `--lr-cutoff 100` for molecules in vacuum | — |

- `--kspace-electrostatics` and `--kspace-dispersion` default to `auto`: PME on
  a fully periodic structure, off otherwise. `none` turns a term off, and a
  method name (`ewald`, `pme`, `p3m`) chooses one.
- The 10 Å cutoff and the fit to a small cell apply only with both terms in
  k-space and no `--lr-cutoff`. With real-space dispersion, the cutoff stays at
  12 Å (or what you set), so dispersion is not cut short.
- With k-space electrostatics, an unset `--kspace-smearing` is 2 Å and an unset
  `--kspace-spacing` 1.55 Å (1 Å below a 10 Å cutoff), or 0.775 × a smearing you
  give.
- `eval` turns PME on only when every structure in the file is fully periodic,
  and never fits the cutoff to a small cell (it sums over the periodic images).
- In a settings file, a k-space key left out is `auto`; `none`, or an explicit
  `null`, turns the term off.
- A slab or wire (`pbc=[True, True, False]`) is refused as a usage error (exit
  2) before anything starts.

The log says what was chosen, for example:

```text
Long range: PME electrostatics and PME dispersion (the default for periodic structures); 10 Å cutoff, 2 Å smearing and 1.55 Å mesh spacing
```

`tune-ewald` and the Python calculators keep their own defaults: a 12 Å cutoff
with no k-space, and 4 Å smearing and 2 Å spacing when k-space is asked for.

### Input and output

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `--input_file` | — | Geometry file in any format that [`ase.io.read`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) can read. |
| `--output`, `--output_file` | `<input>_<ensemble>.xyz` | Trajectory file, `.hdf5` or `.xyz` (suffix `_nvt_langevin` or `_opt` for those commands). |
| `--log-file` | `<output>` with a `.log` extension | Log file; with `--settings`, it is named after the YAML file. |
| `--overwrite` / `--no-overwrite` | `--no-overwrite` | Replace an existing output; a restart appends to an existing EXTXYZ (see [trajectory files](md.md#relaxation-output-and-overwriting)). |

### K-space

| Flag | Default | Meaning |
|---|---|---|
| `--kspace-electrostatics` | `auto` | `auto`, `none`, `ewald`, `pme` or `p3m`. |
| `--kspace-smearing` | `2.0` | Smearing width, Å. |
| `--kspace-spacing` | `1.55` | Mesh spacing, Å (1 below a 10 Å cutoff; 0.775 × a given smearing). |
| `--kspace-interp-nodes` | `4` | Interpolation nodes for PME/P3M. |
| `--kspace-npt-cycles` | every 100 cycles | Under NPT, rebuild the mesh for the current box every N cycles. |
| `--kspace-dispersion` | `auto` | `auto`, `none`, `ewald` or `pme`. |
| `--kspace-dispersion-smearing` | `2.0` | Smearing width for dispersion, Å. |
| `--kspace-dispersion-spacing` | `3.0` | Mesh spacing for dispersion, Å. |
| `--kspace-dispersion-interp-nodes` | `4` | Interpolation nodes for dispersion. |

The MD commands and `opt` take all of them, and `eval` all but
`--kspace-npt-cycles`. `tune-ewald` takes the electrostatics flags without
`auto` and `none`, with 4 Å and 2 Å defaults. See
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

Atom constraints, trajectory subsetting, flexible-cell NPT, the relaxation's FIRE budget and enhanced sampling have no MD flag; set
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
so3lr opt --input molecule.xyz --force-conv 0.05 --lr-cutoff 100
```

`--min-cycles` × `--min-steps` is a hard limit: if it is reached before
`--force-conv`, a warning is logged and the last positions are kept. The output
is an EXTXYZ trajectory with one frame per cycle. See the
[JAX-MD FIRE docs](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html#jax_md.minimize.fire_descent).

The live display and the [journal](terminal-ui.md#run-journal) show the latest
step, potential energy and maximum force (`Fmax`, in eV/Å). The result says
whether the force criterion was met or the step budget ran out; reaching the
step budget is not reported as convergence. `so3lr show <output-directory>`
shows the same later.

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

`tune-ewald` also accepts `--engine` (`auto`, `flash` or `standard`: PME tunes
on flash whenever flash takes the structure), `--jit-compile` /
`--no-jit-compile` and `--settings`, which reads the shared keys and the
[search ranges](settings.md#tune-ewald-search-ranges) from YAML. Its log
defaults to `<input>_tune-kspace.log`. It prints a ready-to-run MD command with
the recommended settings. For the `cutoff` modes, that command and the settings
keys turn k-space off explicitly (`--kspace-electrostatics none
--kspace-dispersion none`), since a periodic structure would otherwise get PME.

## The log file

Every MD run writes a text log. For an MD subcommand it defaults to the output
filename with a `.log` extension; for `--settings` it defaults to the YAML
filename with a `.log` extension. `--log-file` or the YAML `log_file` key can
choose another location. The header records the settings of the run:

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

With k-space on, the header lists the method, smearing and mesh, and a
`Long range: …` line says which [defaults](#long-range-defaults) were chosen for
the structure. It continues with the full merged settings (which can be pasted
back into a settings file), the JAX backend and devices, and the
[engine](md.md#md-engines) used. The model is logged under the name you gave it, so pass a dated name such
as `so3lr-tauri-m-2606` to identify the release.

Please attach the header when reporting a problem.

Each log opens with the SO3LR banner and a line naming the version and start
time, e.g. `◆ SO3LR TAURI · v0.2.0 · 2026-11-13 14:02`. Use different
`log_file` paths for concurrent jobs. The log is a complete plain-text record
whatever `--display`, `--verbose`, `--quiet` or `--no-color` say; redraws and
colour codes are not written into it. MD, optimization and training runs also
write a [journal](terminal-ui.md#run-journal) next to it.

When a run fails, its traceback goes to the log, and the terminal shows the
error message and the log's path (`Full traceback: …`). When the traceback
cannot be confirmed in the log (for example, the log could not be written), the
full traceback is printed instead.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Normal command completion, help, leaving guided setup, or successful snapshot rendering. An optimization may have exhausted its step budget without converging. |
| `1` | The run failed; the error is printed to stderr. |
| `2` | Usage or settings error: unknown flag or model, bad choice, invalid `--settings` YAML, run options without the required input, a slab or wire structure, or a `so3lr resume` that cannot continue. Nothing is computed. |

A bare subcommand such as `so3lr nvt` prints help and exits zero. Supplying run
options without the required input, such as `so3lr nvt --temperature 300`, exits
two. Invalid MD settings files exit two as well; an unknown key only warns and
is dropped (see [Settings file](settings.md)). Runtime failures exit one. An optimization exit of zero means execution
completed; inspect its force-criterion result before treating it as converged.

On `SIGTERM` or `SIGHUP` (`scancel`, a time limit), a run, in a terminal or a
batch job, records the interruption in its journal, prints the
[resume hint](terminal-ui.md#batch-output) and restores the terminal, then
ends by that signal, so the exit status is the signal's. A second signal ends it
at once. Guided setup and `show` restore the terminal the same way. Custom or
ignored signal handlers are left alone. Native numerical code can delay the
first handler; `SIGKILL` ends an unresponsive process without any cleanup.

So you can chain commands in the shell or with scheduler dependencies:

```shell
so3lr opt --input geometry.xyz --output min.xyz \
  && so3lr nvt --input min.xyz --output nvt.xyz --md-cycles 1000
```

## Running on a cluster

One simulation runs on one GPU, except that the flash engine can spread it
over several GPUs of a node with `--flash-devices` (see
[MD engines](md.md#flash-options)). A folder made by `so3lr start` can be
submitted as it is, with `sbatch run.sh` from the folder (add `#SBATCH` lines
to its `run.sh`; see [Run bundles](terminal-ui.md#run-bundles)). Otherwise:

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
checkpoint. With a settings file that sets `restart_save_path`,
`so3lr resume --or-start` starts the run in the first job and continues it in
the next ones, up to the original `md_cycles` (see
[Resuming a run](#resuming-a-run)):

```bash title="chain.sbatch"
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=24:00:00

srun so3lr resume --or-start settings.yaml
```

With flags, the script checks that the checkpoint exists first:

```bash title="chain-flags.sbatch"
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

With flags, each job runs `--md-cycles` more cycles and appends to `traj.xyz`
(an `.hdf5` output needs a new path per job; `so3lr resume` picks one itself). Set `--save-buffer` so checkpoints are written
well within the walltime: a killed job loses everything since the last one. See
[restarts](md.md#restarts).

### Environment variables

These JAX/XLA variables are useful on shared nodes; the last rows are SO3LR's
own.

| Variable | Use |
|---|---|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of GPU memory JAX may take (default `0.75`, raised by SO3LR to `0.97`, or `0.90` with `--flash-devices`, when flash may run and it is unset); lower it to share a GPU. |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Allocate memory on demand, which suits a shared GPU but is slower. |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to the process. |
| `JAX_PLATFORMS=cpu` | Force CPU even when a GPU is available. |
| `SO3LR_CUDA_CACHE` | Where flash's [CUDA kernels](md.md#cuda-kernels) are built (default `~/.cache/so3lr/flash_cuda`). |
| `SO3LR_NVCC`, `CUDA_HOME` | Locate `nvcc` for the CUDA kernels when it is not on the `PATH`. |
| `SO3LR_JOURNAL=0` | Write no [run journal](terminal-ui.md#run-journal). |
| `SO3LR_NO_UPDATE_CHECK` | Any value other than empty or `0` turns off the [release notice](terminal-ui.md#release-notices); it is also off when `CI` is set. |
| `NO_COLOR` | No terminal colours, like `--no-color`. |

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
