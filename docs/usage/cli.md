# Command line

SO3LR provides one command-line interface for geometry optimization,
molecular dynamics, evaluation, training and parameter tuning.

## Commands

| Command | What it does |
|---|---|
| `so3lr start` | Guide MD setup, review, then start now or save a run bundle |
| `so3lr show PATH` | Browse local MD and optimization journals while jobs run, or inspect completed runs |
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

Use `--settings` without an MD subcommand. Display controls below are CLI
options, rather than simulation settings in the YAML file.

## Terminal display

MD runs show a compact live display in an interactive terminal. It reports
the current stage, model, engine, hardware, simulation progress and measured
quantities, with temperature and **total energy** as the initial graphs. NPT also
shows volume. Stacked plots have aligned time axes and update at accepted cycle
boundaries, whether started with full command flags, `--settings settings.yaml`,
or guided setup. Full commands and YAML runs never open the setup wizard. Speed shows both
`ns/day` and `s/atom/step`; the
latter divides measured seconds per step by the total simulation atom count.
Mixed precision is labeled `mixed (float32 model; float64 dynamics)`.
Once MD samples arrive, a three-row header shows the run state and details, a
progress bar with completed steps and simulated time, and `MD elapsed`, ETA and
throughput; the plots fill the remaining rows. Colors follow one palette: amber
marks SO3LR, focus, progress, the temperature trace and the Running state;
energy is cyan and NPT volume violet. `--no-color` and `NO_COLOR` remove the
colors, and terminals without Unicode get ASCII glyphs.
During initialization, compilation and other long stages, the Tauri pulse and
a clock counting time since launch refresh twice per second. The star expands and recedes in
place over five seconds; terminals without Unicode use an ASCII pulse.
This refresh uses a frozen text snapshot and a wall
clock, without querying model arrays or running numerical work. Measured values
and graphs remain at their latest reporting point.
During MD, `e` changes the energy plot and up/down changes plot pages in a
short terminal. Keys are handled at reporting boundaries. **Esc** hides the
plots without stopping MD; pressing Esc again keeps them hidden. `q` toggles between
plots and compact status; press it again to restore the graphs with the full
recorded history. MD continues in either view; Ctrl+C still interrupts MD.
The complete settings and per-cycle data still go to the text
log. With redirected stderr (including ordinary scheduler logs) or `TERM=dumb`,
the default is plain text output. Redirecting stdout alone does not disable the
live display. A `.sh` script run directly in a terminal can still show it;
use `--display plain` to explicitly request ordinary text logging.

Put these global options **before the subcommand**:

| Flag | Default | Meaning |
|---|---|---|
| `--display` | `auto` | `auto` uses the live MD/optimization display when stderr is a terminal; `rich` explicitly requests it on a terminal; `plain` uses ordinary text logging. Redirected `rich` output falls back to plain with a warning (unless quiet). |
| `--verbose`, `-v` | off | Also print detailed SO3LR diagnostics during the live display. Plain output already includes them. |
| `--quiet`, `-q` | off | Suppress informational SO3LR log messages and the live display; keep warnings and errors. |
| `--no-color` | off | Disable terminal colors; the `NO_COLOR` environment variable does the same. |
| `--version` | — | Print the installed package version and exit. |

```shell
so3lr --display plain nvt --input geometry.xyz --temperature 300
so3lr --verbose --settings settings.yaml
so3lr --no-color nvt --input geometry.xyz
```

`--settings` is a root command alternative to a simulation subcommand. Combining
it with a subcommand is a usage error; the settings file is never silently ignored.
`--verbose` and `--quiet` cannot be combined. Color controls do not disable
live updates; use `--display plain` for output with no terminal redraws.
These options leave the saved log and simulation settings unchanged.

The live display covers `nvt`, `nvt-langevin`, `npt`, `nve`, MD launched
with `--settings`, and `opt`, including the standard and Flash engines.
Optimization shows its latest reported step, potential energy and maximum
force instead of MD plots. Its result states whether the force criterion was
met or the step budget was reached. Evaluation, training and tuning retain
their existing output and progress bars.

## Logs and reproducibility

MD and optimization logs retain their timestamped diagnostics and numeric
tables, with additional run identity and final summary blocks. A run ID links
the log to its journal and resolved settings, including when several invocations
append to a log. Model/backend updates use the compact label
`Model: ...; backend: ...`. Full run IDs appear in the `BEGIN RUN`, `RUN SUMMARY`
and `END RUN` markers, journals and artifact paths. Use different `log_file`
paths for concurrent jobs.

The final summary separates execution status from scientific outcome. For
example, an optimization can finish its step budget without meeting its force
criterion. It also reports completed MD steps and physical time, final measured
quantities with units, warnings/errors, wall time, the last successful checkpoint,
and saved artifact paths. Unavailable information is not invented.

MD and optimization failures retain the exception type and cause chain in
the journal. The terminal error is shortened only after the
traceback is confirmed in the current text log. If setup or log writing fails,
the full terminal traceback remains available. Unexpected errors from other
commands also keep their normal traceback. When preservation is verified, the
terminal prints `Full traceback:` with the absolute log path. Multiline errors
use a concise first line in the terminal; the complete diagnostic stays in the
log and journal. A failed or unverifiable log write keeps the full terminal traceback.

A failed or interrupted MD run ends with its last saved checkpoint and how to
continue from it, for example:

```text
Last checkpoint: step 450,000 · restart.npz
Resume: --restart-load restart.npz (YAML: restart_load_path: restart.npz); relaxation is skipped and the trajectory is appended.
To finish the original length, also set --md-cycles 5500 (YAML: md_cycles: 5500); md_cycles counts from the checkpoint.
```

A restart skips the initial relaxation and appends to an EXTXYZ trajectory; HDF5
output needs a new output path. It runs `md_cycles` more cycles from the
checkpoint, so the hint gives the cycle count that ends at the original length. Without `restart_save_path` (`--restart-save`)
no checkpoint exists, and the message says how to make runs resumable. Ctrl+C
appears on screen as `Interrupted (Ctrl+C)`; the journal keeps the exact
exception record.

Plain and batch output (for example a Slurm `.out` file) adds one progress line
at every checkpoint, and at least every ten minutes:

```text
Progress 45.0% · step 450,000/1,000,000 · 2.31 ns/day · ETA 1d 02:13:00
```

Throughput and ETA use wall time since the first MD sample, including output.
After the journal path, each run also logs `Watch live: so3lr show <journal>`,
which opens that run directly.

Warnings logged through SO3LR, including worker-thread warnings, reach the
enabled journal and terminal without waiting for the next MD cycle. Attached plot
state still updates on its owner thread. While the live display is active,
Python output written to the current `sys.stderr`, including ordinary warnings
and newly attached logging handlers, is routed through Rich. Third-party handlers
that already hold another stream, native writes to file descriptor 2 and
`sys.__stderr__` are not redirected. Libraries keep their own handlers, and the
journal does not claim to capture all library diagnostics.

By default, each MD or optimization invocation saves
`<log-stem>.<run-id>.settings.resolved.yaml` alongside its journal, including
with redirected stderr, `--display plain` or `--quiet`.
It uses ordinary YAML values and input units (`md_dt` in fs), resolves input paths,
and records the selected model, engine and precision when available. To repeat
a run, copy this file, choose fresh output/log/checkpoint paths, and run:

```shell
so3lr --settings repeat.yaml
```

Saved optimization settings include `operation: optimization`; the same command
then uses the optimization command's normal option validation. Ordinary MD YAML
files keep their existing behavior. Saved settings retain overwrite protection.
Input and restart files are referenced, not archived; retain the original input
and checkpoint contents if exact replay is required. Warning/error totals count
logging records, including file-only diagnostics, rather than unrelated Python
warnings printed outside the logger.

## Offline monitoring

By default, every CLI MD or optimization invocation also creates a unique
`<log-stem>.<run-id>.events.jsonl` next to its text log. This happens with
redirected output, `--display plain`,
and `--quiet` too. The journal records run metadata, phases, accepted-cycle
samples or the latest optimization measurements, warnings, successful checkpoints
and completion/failure. Each process owns a separate journal, including
concurrent replicas in one scheduler job. The filename keeps its UUID-based
run identity; it is not a chronological label. The viewer discovers files by
the `.events.jsonl` suffix and orders them by path, so appending samples cannot
move existing runs around. Existing journal filenames continue to work.
These two sidecar files support runs without a live display: the journal
supports offline observation, and the resolved YAML records the replay settings.
Display controls change console output, not artifact creation. To disable both
sidecars for an invocation, set the environment variable to exactly `0`:

```shell
SO3LR_JOURNAL=0 so3lr --settings settings.yaml
```

Unset or any other value keeps the default enabled. This opt-out keeps the text
log, summaries, trajectories, checkpoints and in-memory attached plots. It writes
no metrics JSONL or resolved-settings YAML and is noted in the text log; it is
not reported as a disk failure. `so3lr show` cannot reconstruct that invocation
without its journal.
The numeric table columns and trajectory formats are retained; text logs also
include the identity and final summary blocks described above.

```shell
# In your ordinary .sh script:
so3lr --settings settings.yaml

# In another terminal, on a machine that can read the output directory:
so3lr show path/to/run-directory
so3lr show path/to/experiment-directory
so3lr show path/to/run-directory --once
# From inside the run directory:
so3lr show .
```

The native terminal viewer works offline and uses no W&B integration, account
or server. It reads files only; closing it never sends a signal to the job.
An existing directory works for both running and completed jobs. The viewer
needs a `*.events.jsonl` journal from this CLI; older folders containing only
the ordinary text log are not parsed as graph data. In live mode it checks
for new samples every second by default. `--once` is a human-readable static
snapshot with all plots and rows, without keyboard hints or paging controls.
It exits zero when the snapshot is rendered, even if the recorded run failed,
is stale, or is still being read. Its text is not a machine-readable status API.
Explicit file inputs must be `*.events.jsonl`; passing a text log or trajectory
gives an error and points to journal discovery instead.
Use `j`/`k` or left/right to choose a run, `e` to switch the energy plot, and
**Esc** or `q` to close the viewer, including from the folder overview. Closing
it leaves the simulation running. Temperature and the selected energy are stacked vertically;
total energy is selected initially.
For a folder containing several runs, the first view is an overview with
each run's reported state, progress and age since its last journal update.
Age stays visible in narrow views; other columns adapt to the available width.
Terminals of at least 100 columns (120 with host details) draw a progress bar
next to each run's percentage. Runs that ended with Ctrl+C or a termination
signal show as **Interrupted** and are counted apart from failures.
When present, the recorded host and Slurm job ID help identify batch runs.
Individual views retain the full origin details. When optimization and MD share a folder, the
overview labels OPT and MD explicitly. Optimization progress counts its own
steps; its temperature and MD total-energy fields remain unavailable.
Left/right steps through the individual runs and back to the overview;
`o` returns directly to the overview. Up/down pages through a long run list.
NPT runs add a volume plot in Å³ below them, using the recorded physical cell.
Short terminals show complete plots on pages: use up/down to change pages;
the footer shows the current page and controls. Full snapshots show all plots.
These plots use recorded samples, with no extra model evaluations. Temperature
uses the requested thermostat target for NVT, NPT and Langevin dynamics.
Reference guides are solid, muted lines behind the trajectory. The exact value
appears on the plot's heading row (below the plot in narrow terminals); line
positions use terminal-character rows.
NVE has no thermostat: its temperature guide is a measured reference, as are
the guides for each energy plot and NPT volume. These references are means of
samples in the latter half of the recorded time span, not target values.
For long runs, bounded time bins retain sums and counts independently of graph
decimation; a reference marked approximate uses an estimated contribution
from the bin crossing the midpoint. Time axes share the same bounds and aligned
axis columns, with intermediate numbered ticks and fewer labels in narrow
terminals. Each property keeps its own units and value scale. Conserved
energy is absent for Langevin dynamics. Cycle throughput shows `ns/day` and
`s/atom/step` (seconds per step divided by the total simulation atom count),
excluding startup and output overhead. Unknown timing or atom counts are
omitted rather than estimated.

For an optimization journal, `show` displays the latest recorded optimization
cycle: step count, potential energy, maximum force and any force target. A
completed invocation does not imply convergence. The view distinguishes
**Force criterion met**, **Step budget reached; force criterion not met**, and
**Step budget finished (no force target)**.

| Show option | Default | Meaning |
|---|---|---|
| `PATH` | current directory | One journal, or a directory searched up to three subdirectory levels. Discovery is bounded to 128 files and 512 directories; point at a narrower directory for large experiments. |
| `--once` | off | Print a human snapshot without interactive controls and exit zero on successful rendering; this does not establish simulation success or liveness. Automatic for noninteractive input/output. |
| `--interval` | `1.0` | Seconds between journal reads. |

Graphs use **time within the current invocation**, starting at zero after a
restart. Older checkpoints do not establish an unambiguous absolute physical
time if the cycle size or timestep changed. Long histories are reduced only
in the viewer, preserving extrema; the full samples remain in the journal.

Data arrives at accepted MD cycle boundaries, so large `md_steps` can leave a
chart unchanged for minutes. The displayed state is the last state explicitly
reported by the producer, not a scheduler or process-liveness check. A run without
a final event is marked as potentially stale when no update has arrived for
more than 10 minutes or 20 times its median recorded sample interval, whichever
is longer. This is a freshness warning, not proof that a process has died.
Missing timestamps remain unknown. A viewer still catching up with unread
journal bytes says it is reading history rather than presenting an early state
as current. Age and checkpoint information help distinguish recent work from saved work.
An externally terminated process can leave its last lifecycle event as Running:
plain and batch runs retain native signal handling, and `SIGKILL` cannot record
a final event. A journal write
failure warns once and disables metrics recording while the calculation
continues; do not use the viewer as the sole check of a job's health.

## Guided setup

Run `so3lr start` to follow **1/3 Structure**, **2/3 Dynamics**, and **3/3 Review
and start**. Choose one of the bundled examples, browse directories with the
arrow keys and Enter, or paste a file path. Dynamics, model and review choices
also use keyboard menus.

Three small examples ship with the package, so they are available in any
installation: an Ala4 peptide (42 atoms) and DHA (56 atoms), both in the gas
phase, and a box of 64 water molecules (192 atoms, periodic, 12.4 Å). The water
box is too small for the usual 12 Å cutoff, so setup proposes a shorter cutoff
with PME (see below).

For your own multi-frame XYZ/EXTXYZ input, choose the frame explicitly. The
selected frame is exported into the new run bundle; the source is unchanged.
If the input includes velocities, choose whether to keep them or initialize
velocities at the requested temperature. The review shows the choice.

The first pass asks only the essentials: ensemble, model, temperature (and
pressure for NPT), duration, how often to save a trajectory frame (in ps),
total charge, unpaired electrons and the output directory. The duration is
rounded up to whole frames, and setup says so (for example
`Duration rounded up to 5.05 ps (101 whole frames)`); the frame interval must be
a whole number of timesteps. When the XYZ comment line records `charge` or the
spin multiplicity, those values become the defaults. Setup reads the
multiplicity (2S+1) from `multiplicity`, or from `spin` as in OMol25 files, the
same rule `so3lr eval` uses. An impossible combination, such as a neutral
radical with no unpaired electron, is explained, and setup asks before keeping it. Timestep, random seed, initial
relaxation and the long-range cutoff start from the defaults that setup prints.
The default output directory is `so3lr-run`, or the next free name
(`so3lr-run-2`, ...).

The model menu presents **M (default)**, **S (faster)** and **L (recommended
for reactions)** first. Open **Legacy models** to see the four older releases;
exports retain each model's canonical name. The review explains mixed precision
as float32 model calculations with float64 dynamics. For periodic cells, the
cutoff must fit the minimum-image neighbor list: twice the larger of the
model's short-range cutoff and `lr_cutoff` must fit the minimum cell height.
For **NPT**, setup checks against 97% of that height, reserving the backend's
initial 3% contraction margin. It applies the same check to a cutoff you enter.
Fixed-cell ensembles keep the ordinary half-cell bound. During MD, each
evolving cell is checked against the physical half-cell bound. Later
contraction can still invalidate a cutoff, and Flash has additional cell-grid
requirements; the runtime checks both.

When the usual cutoff does not fit, setup proposes a shorter cutoff with PME
electrostatics and dispersion. The review and saved YAML include both smearing
and mesh spacing:

| Small-cell proposal | Smearing | Mesh spacing |
|---|---|---|
| PME electrostatics | 2 Å | 1 Å |
| PME dispersion | 2 Å | 3 Å |

The electrostatic mesh is refined with the narrower smearing, retaining the
normal spacing-to-smearing ratio. These are starting settings, not a full
accuracy-tuning result; check convergence for your system. Existing direct
commands and YAML settings retain their defaults and validation.

The review groups the run into System, Dynamics and Output. It shows the
effective run length, trajectory frames, checkpoint cadence and the files that
setup writes and that the run will add. **Save setup and start MD now** is the
first and default review choice. Pressing Enter there writes the structure,
YAML and shell script, then immediately runs MD in this terminal. **Save setup
for later** writes those files only. **Edit settings…** changes one group
(temperature and pressure, duration and frame interval, timestep, charge and
multiplicity, seed and relaxation, cutoff, or output directory) and returns to
the review. Its first entry starts over, keeping every previous answer as the
default. Nothing is written before this explicit review choice. Esc in a
selection menu cancels setup. Cancel or Ctrl+C before export also writes nothing.

The new bundle contains the selected structure, `settings.yaml` and an
executable `run.sh`. A single-frame input is copied byte-for-byte when no
boundary or velocity change is selected; otherwise the reviewed frame is
written as EXTXYZ. The launcher changes to its own directory before running
`so3lr --settings settings.yaml`; activate your SO3LR environment before using
it. Existing output directories are refused. All periodic CLI MD modes require
a positive, axis-aligned orthorhombic cell. NPT additionally requires periodic
boundaries in all three axes. Guided setup does not submit scheduler jobs;
use the generated configuration from your existing submission script.

A bare `so3lr` opens the same setup only when input and output are terminals,
`TERM` is not `dumb`, and no recognized scheduler job environment is present.
Otherwise it prints help. Simulation commands and `--settings` never open
guided setup.
`so3lr start` without a terminal gives a clear error instead of waiting for input.
`so3lr init` remains a hidden compatibility alias; new commands and examples
use `so3lr start`.
Help, drafting a run, and watching journals do not initialize JAX or load weights.

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

MD and `opt` keep the charge and spin you pass, but warn when they contradict
the values recorded in the structure file (`charge`, `multiplicity` or `spin`,
read as in `so3lr eval`) or describe an impossible electron count.

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

The live display and saved journal include the latest completed step, potential
energy and maximum force (`Fmax`, in eV/Å). These values update at the existing
optimization reporting points. Completion of the command and satisfaction of
the force criterion are shown separately; reaching the step budget is not
reported as convergence. Use `so3lr show <output-directory>` to inspect this
summary later.

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

It continues with the full merged settings (which can be pasted back into a
settings file), the JAX backend and devices, and the [engine](md.md#md-engines)
used. The model is logged under the name you gave it, so pass a dated name such
as `so3lr-tauri-m-2606` to identify the release.

Please attach the header when reporting a problem.

The log remains a complete plain text record regardless of `--display`,
`--verbose`, `--quiet` or `--no-color`. Live progress redraws and color codes
are not written into it.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Normal command completion, help, leaving guided setup, or successful snapshot rendering. An optimization may have exhausted its step budget without converging. |
| `1` | The run failed; the error is printed to stderr. |
| `2` | Usage/configuration error: unknown flag or model, bad choice, invalid root `--settings` YAML, or run options supplied without required input. Numerical execution does not start. |

A bare subcommand such as `so3lr nvt` prints help and exits zero. Supplying run
options without the required input, such as `so3lr nvt --temperature 300`, exits
two. MD and optimization YAML validation use the same usage-error status;
unknown MD YAML keys still warn rather than abort. Runtime failures remain
distinct from input validation. An optimization exit of zero means execution
completed; inspect its force-criterion result before treating it as converged.

When a live display or keyboard owns an attached terminal, default `SIGTERM`
and `SIGHUP` handlers unwind the UI and restore terminal input/cursor state before
preserving the native signal exit status. This also covers interactive setup
and `show`. Plain/quiet/batch runs and forced Rich output without a terminal keep
native termination. Custom or ignored signal handlers are left alone. Once
Python handles the first signal, defaults are restored before cleanup so a
second signal can terminate immediately. Before that handler runs, native
numerical code may still delay it; `SIGKILL` remains the way to terminate an
unresponsive process, without UI cleanup.

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
