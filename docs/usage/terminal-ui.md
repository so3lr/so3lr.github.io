# Guided setup and monitoring

`so3lr start` sets up a run through keyboard menus and saves it as a folder
that runs itself. While a run computes, the terminal shows its progress and
graphs, and every MD, optimization and training run keeps a journal that
`so3lr show` can open from another terminal, during the run or after it. None
of this is needed: direct commands and settings files work the same way, with
or without a terminal. The commands and flags are on the
[Command line](cli.md) page.

## Quick tour

```shell
so3lr start                      # choose a task, a structure and settings; review; start or save
sh so3lr-run/run.sh              # run a saved setup (or: cd so3lr-run && sbatch run.sh)
so3lr show so3lr-run             # watch it from another terminal, live or afterwards
so3lr resume so3lr-run           # continue it from its last checkpoint
```

`so3lr start` saves the run in a new folder (`so3lr-run`, or `so3lr-run-2`,
... when that exists) and can start it at once in the same terminal. The
folder's `run.sh` starts the run, or continues it when a checkpoint exists, so
the same script serves the first job and every resubmission.

## Guided setup

`so3lr start` first asks what you would like to do:

| Task | What setup saves | Run later with |
|---|---|---|
| **Run molecular dynamics** (default) | `structure.extxyz`, `settings.yaml`, `run.sh` | `sh run.sh` or `sbatch run.sh` |
| **Optimize a geometry** | `structure.extxyz`, `run.sh` calling `so3lr opt` | `sh run.sh` |
| **Evaluate a dataset** | `run.sh` calling `so3lr eval` (the dataset is referenced, not copied) | `sh run.sh` |
| **Watch runs** | nothing; opens [`so3lr show`](#so3lr-show) on a folder | — |
| **Other tasks** | nothing; lists the `finetune`, `train` and `tune-ewald` commands | — |

Nothing is written before the review, and drafting a run loads no model and
does not start JAX.

### Molecular dynamics

Setup follows **1/3 Structure**, **2/3 Dynamics** and **3/3 Review and start**.
Choose a bundled example, browse directories with the arrow keys and Enter, or
paste a path to an `.xyz` or `.extxyz` file. For a multi-frame file, choose the
frame; it is copied into the new folder and the source is left unchanged. If
the input has velocities, choose whether to keep them or draw new ones at the
requested temperature.

The first pass asks only the essentials: ensemble, model, temperature (and
pressure for NPT), duration, how often to save a trajectory frame (in ps),
total charge, unpaired electrons and the output folder. The duration is rounded
up to whole frames, and setup says so (for example
`Duration rounded up to 5.05 ps (101 whole frames)`); the frame interval must be
a whole number of timesteps. Charge and multiplicity default to the values in
the XYZ comment line: `charge`, and the multiplicity (2S+1) from `multiplicity`,
or from `spin` as in OMol25 files, the same rule `so3lr eval` uses. An
impossible combination, such as a neutral radical with no unpaired electron, is
explained, and setup asks before keeping it. Timestep, random seed, the initial
relaxation and the long-range cutoff start from the defaults that setup prints.

The model menu lists **M (default)**, **S (faster)** and **L (recommended for
reactions)** first; **Legacy models** opens the older releases. The saved
settings use each model's dated name.

The long-range settings follow the cell, with the same rule as the
[command line](cli.md#long-range-defaults):

| Structure | Cutoff | Electrostatics | Dispersion |
|---|---|---|---|
| Isolated (no periodic cell) | 100 Å | real space | real space |
| Periodic, cell fits a 10 Å cutoff | 10 Å | PME, 2 Å smearing, 1.55 Å mesh | PME, 2 Å smearing, 3 Å mesh |
| Periodic, smaller cell | 0.45 × the cell height, rounded down to 0.1 Å, at least the model's short-range cutoff | PME, 2 Å smearing, 1 Å mesh | PME, 2 Å smearing, 3 Å mesh |

These are starting settings, not a tuning result; check convergence for your
system. A cutoff you enter must fit the cell: twice the larger of the model's
short-range cutoff and `lr_cutoff` must fit the minimum cell height. For NPT,
setup checks against 97% of that height, leaving room for the box to shrink at
the start; the runtime still checks every later cell. Periodic MD needs a
positive, axis-aligned orthorhombic cell, and NPT a cell periodic along all
three axes.

The review groups the run into System, Dynamics and Output. It shows the run
length, the trajectory frames, how often a checkpoint is written, and the files
that setup writes and that the run will add. The choices are:

- **Save setup and start MD now** (the default): writes the folder, then runs MD
  in this terminal.
- **Save setup for later**: writes the folder only.
- **Edit settings…**: changes one group (temperature and pressure, duration and
  frame interval, timestep, charge and multiplicity, seed and relaxation,
  cutoff, or output folder) and returns to the review. Its first entry starts
  over, keeping every answer as the default.
- **Cancel**: writes nothing.

An existing output folder is refused.

### Optimization and evaluation

**Optimize a geometry** asks for the structure, the model, a force threshold
(default 0.05 eV/Å), a step budget (default 1,000), charge and multiplicity, and
the output folder (`so3lr-opt`). The optimization stops at whichever comes
first. The long-range settings follow the table above.

**Evaluate a dataset** asks for an `.xyz`/`.extxyz` dataset with reference
energies and forces (the bundled example, the file browser, or a pasted path),
the model, the long-range cutoff (10 Å for periodic data, 100 Å otherwise) and
the output folder (`so3lr-eval`). The run writes `predictions.extxyz` and
`eval.log`. The bundled example holds 10 SAPT10k dimers (20 frames: each dimer
followed by its separated pair); for such files the evaluation reports binding
energies against the SAPT reference (see
[Binding energies](../advanced/binding-energies.md)).

Both review with **Save setup and start now**, **Save setup for later** and
**Cancel**. The saved `run.sh` holds the complete `so3lr opt` or `so3lr eval`
command, so it can be read and edited.

### Example structures

Three small examples ship with the package, so they are available in any
installation: an Ala4 peptide (42 atoms) and DHA (56 atoms), both without a
cell, and a box of 64 water molecules (192 atoms, periodic, 12.4 Å). The water
box is too small for the 10 Å cutoff, so setup proposes 5.5 Å with PME
electrostatics and dispersion and a 1 Å mesh.

### Keys

| Key | Action |
|---|---|
| Up/down, Enter | Move and select |
| Esc or Backspace | Back one step: from a task's first question to the task menu, from a review to its settings; at the task menu, close setup |
| Ctrl+C | Cancel; nothing is written before the review choice |

### Without prompts

`so3lr start --input STRUCTURE` writes the MD folder that the guided setup
would propose, with no prompts, for scripts and agents:

```shell
so3lr start --input box.xyz --ensemble npt --temperature 300 --duration 100
so3lr start --input box.xyz --ensemble npt --print     # print settings.yaml; write nothing
```

It prints the folder and how to start it. `--print` prints the `settings.yaml`
instead of writing a folder. Every option of the guided MD setup has a flag
(`so3lr start --help` lists them):

| Flag | Default | Meaning |
|---|---|---|
| `--input` | — | `.xyz` or `.extxyz` structure. |
| `--frame` | — | Frame of a multi-frame file, 1-based; required for one. |
| `--ensemble` | `nvt` | `nvt`, `nvt-langevin`, `nve` or `npt`. |
| `--model` | `so3lr-tauri-m-2606` | Model name or path. |
| `--temperature`, `--pressure` | `300`, `1.0` | K and bar (pressure for NPT only). |
| `--duration` | `5` | Length in ps, rounded up to whole frames. |
| `--frame-interval` | `0.05` | ps between saved frames, a whole number of timesteps. |
| `--dt` | `0.5` | Timestep in fs. |
| `--total-charge`, `--spin` | from the file, else `0` | Charge and unpaired electrons. An impossible combination gives a warning. |
| `--seed` | `52` | RNG seed. |
| `--relax` / `--no-relax` | `--relax` | Relax the geometry before MD. |
| `--lr-cutoff` | from the cell | 10 Å for a periodic structure (fitted to a small cell), 100 Å for an isolated one. |
| `--keep-velocities` | off | Keep the velocities stored in the file. |
| `--output-dir` | `so3lr-run`, ... | New run folder. |
| `--print` | off | Print the settings instead of writing a folder. |

Other options without `--input` are a usage error. A bare `so3lr start` needs an
interactive terminal and says so without one. A bare `so3lr` opens the same
setup only in an interactive terminal (`TERM` not `dumb`) outside a scheduler
job; otherwise it prints help. Simulation commands and `--settings` never open
guided setup. `so3lr init` is a hidden alias of `so3lr start`.

## Run bundles

An MD folder holds the reviewed frame as `structure.extxyz` (the input copied
byte for byte when nothing in it changed), a `settings.yaml` with every setting
of the run, and an executable `run.sh`. The paths in `settings.yaml` are
relative to the folder:

| File | Written by |
|---|---|
| `structure.extxyz`, `settings.yaml`, `run.sh` | setup |
| `trajectory.extxyz` | the run |
| `trajectory_opt.xyz` | the relaxation before MD |
| `run.log` and `run.<id>.events.jsonl` | the run: [log](cli.md#the-log-file) and [journal](#run-journal) |
| `restart.npz` | the run's checkpoint, every `save_buffer` cycles and at the end |

`run.sh` changes to its own folder and runs

```shell
exec so3lr resume --or-start .
```

so a copied or moved folder runs itself. It starts the run when there is no
checkpoint yet, and otherwise continues it from `restart.npz` for the rest of
its length (see [Resuming a run](cli.md#resuming-a-run)). Activate your SO3LR
environment first. The script also works under `sbatch run.sh`: Slurm runs a
copy from its spool directory, and the script then uses the folder's absolute
path, which setup wrote into it. Add `#SBATCH` lines at the top as your cluster
requires. Resubmitting the same `run.sh` after a time limit continues the run,
and once the run is complete it says so and exits.

Optimization and evaluation folders hold a `run.sh` that runs the saved
`so3lr opt` or `so3lr eval` command (`exec so3lr opt ...`) from the folder.

## Live display

In an interactive terminal, MD shows its stage, model, engine, hardware and
progress, with temperature and **total energy** graphs (NPT adds volume).
Graphs share aligned time axes and grow as each MD cycle is accepted. The
three-row header shows the run state, a progress bar with completed steps and
simulated time, and the elapsed time, ETA and throughput in `ns/day` and
`s/atom/step` (seconds per step divided by the atom count). Mixed precision is
labelled `mixed (float32 model; float64 dynamics)`. Colours follow one
palette: amber for SO3LR, progress and temperature, cyan for energy, violet for
volume. Terminals without Unicode get ASCII glyphs.

During initialization, compilation and other long stages, a pulsing star and a
clock since launch refresh twice a second. Once a rate is measured, the same
line adds an ETA, e.g. `Running · 00:21 since launch · ETA 07:22`; the rate
leaves out the first unit of work, which usually includes compilation. An
optimization with a force target shows `ETA ≤ …`, since it may converge sooner.
Measured values change only at reporting points, so a large `md_steps` can
leave a graph unchanged for a while. Graph views redraw at most once a second
and text views every 0.25 s; a key press redraws at once.

Every command that computes uses the display: the four MD commands, MD from
`--settings`, `opt`, `eval`, `tune-ewald`, `finetune` and `train`, on both
engines. Each shows its stage, a progress bar with elapsed time, ETA and rate
(steps, structures, candidates or epochs), and a results table: errors against
the labels, binding energies and timing for `eval`; the recommended settings
and a ready-to-run MD command for `tune-ewald`; the latest validation loss and
errors for training. Optimization adds potential energy and maximum force (log
axis, force target dashed) once three cycles are recorded, and training adds
validation loss and force MAE (the energy MAE when the data has no forces)
once three validations are recorded. Its result says whether the force
criterion was met or the step budget ran out.

The complete settings and per-cycle data still go to the text log, which never
contains redraws or colour codes.

### Keys

| Key | During MD |
|---|---|
| `e` | Change the energy plot |
| Up/down | Change the plot page in a short terminal |
| Esc | Hide the graphs; MD continues |
| `q` | Switch between graphs and compact status; switching back shows the whole history |
| Ctrl+C | Interrupt MD |

Keys are handled at reporting points.

### The closing line

Every run ends with one line, in plain and batch output too:

```text
✓ Evaluation finished in 00:03 · 20 structures · binding-energy MAE 0.94 kcal/mol
! Optimization finished in 00:14: force target not met · max force 0.00635 > 0.005 eV/Å
■ MD interrupted at step 220 after 00:15
✕ Tuning stopped after 00:01 · ValueError: …
```

Green ✓ means the run finished and met its goal; amber `!` that it finished
without it (an unmet force target, no k-space candidate within the accuracy);
amber ■ an interruption; red ✕ an error. An interrupted or failed MD run then
prints its [last checkpoint](#batch-output) and how to resume. `--quiet` prints
no closing line. An exit code of 0 from `opt` means the optimization ran, not
that it converged.

### Display options

| Flag | Default | Meaning |
|---|---|---|
| `--display` | `auto` | `auto` uses the live display when stderr is a terminal; `rich` asks for it (on a redirected stderr it falls back to plain, with a warning); `plain` writes ordinary log lines. |
| `--verbose`, `-v` | off | Also print detailed diagnostics during the live display (plain output has them already). |
| `--quiet`, `-q` | off | Only warnings and errors; no live display. |
| `--no-color` | off | No colours; the `NO_COLOR` environment variable does the same. |

They may go before or after the subcommand, and they change only what the
terminal shows, not the run or its files:

```shell
so3lr --display plain nvt --input geometry.xyz
so3lr nvt --input geometry.xyz -q
so3lr --verbose --settings settings.yaml
```

`--verbose` and `--quiet` cannot be combined. With a redirected stderr (as in
most scheduler jobs) or `TERM=dumb`, the display is plain; redirecting stdout
alone keeps the live display.

### Release notices

In an interactive terminal, SO3LR checks GitHub at most once a day for a newer
release and, after a successful command, prints one line such as
`SO3LR 0.3.0 is available (you have 0.2.0)` with the upgrade command. The check
runs in the background with a short timeout, caches its answer in
`~/.cache/so3lr/` (or `$XDG_CACHE_HOME/so3lr/`), and sends no user information.
It never runs without a terminal, with `--quiet`, in a scheduler job, or when
`CI` is set. Set `SO3LR_NO_UPDATE_CHECK` to any value other than empty or `0` to
turn it off.

## Run journal

Every MD, optimization and training run writes a journal next to its text log,
`<log-stem>.<run-id>.events.jsonl`, also with redirected output,
`--display plain` or `--quiet`. It records the run's settings and host, its
phases, the accepted MD cycles (or the latest optimization step, or each
validation), warnings, checkpoints, and how the run ended. Each process writes
its own journal, so concurrent replicas in one job do not collide; the run ID is
random, not a date. `so3lr show` reads it, and the log names the journal and the
command to watch it (`Watch live: so3lr show …`).

`finetune` writes `<workdir>/finetuning.<id>.events.jsonl` by default.
Evaluation and `tune-ewald` keep no journal. To write none, set
`SO3LR_JOURNAL=0`:

```shell
SO3LR_JOURNAL=0 so3lr --settings settings.yaml
```

Any other value keeps it. The text log, trajectories, checkpoints and live
graphs stay; `so3lr show` then has nothing to read for that run. If a journal
write fails, SO3LR warns once and continues without it.

## `so3lr show`

```shell
so3lr show path/to/run-folder          # one run, or an overview of several
so3lr show path/to/experiments         # searched up to three folders deep
so3lr show so3lr-run/run.<run-id>.events.jsonl   # one journal
so3lr show . --once                    # print one snapshot and exit
so3lr show . --json                    # machine-readable snapshot
```

`show` reads the journals only. It needs no W&B account or server, sends no
signal to the job, and closing it leaves the run running. It works on any
machine that can read the output folder, while the run computes or after it.
Folders with only a text log (no journal) are not read.

| Option | Default | Meaning |
|---|---|---|
| `PATH` | current folder | A journal, or a folder searched up to three levels deep (at most 128 journals and 512 folders; point at a narrower folder for large experiments). |
| `--once` | off | Print one snapshot with all plots and exit; automatic when output is not a terminal. |
| `--json` | off | Print one machine-readable snapshot and exit (see [below](#json-snapshot)). |
| `--interval` | `1.0` | Seconds between journal reads. |

A text log or trajectory given as `PATH` is refused with a pointer to its
journal.

### Views and keys

A folder with several runs opens on an overview: each run's state, progress
and the age of its last journal update, with the host and Slurm job ID when
recorded and a progress bar in terminals at least 100 columns wide. Runs are
labelled `MD`, `OPT` or `TRN`; MD progress counts steps, optimization progress
its own steps, and training progress finished epochs. Runs stopped by Ctrl+C or
a termination signal show as **Interrupted**, apart from failures.

| Key | Action |
|---|---|
| `j`/`k` or left/right | Next or previous run, cycling through the overview |
| `o` | Back to the overview |
| `e` | Change the energy plot |
| Up/down | Page through plots or a long run list |
| Esc or `q` | Close the viewer |

A run's view stacks temperature and the selected energy (total energy first),
with volume for NPT, and the optimization or training graphs for those runs.
Reference lines are muted: the thermostat target for NVT, NPT and Langevin
temperature, and for NVE temperature, energies and volume the mean over the
second half of the recorded time, labelled `Second-half mean`. Time starts at
zero in each invocation, including restarted segments. Long histories are
thinned for drawing only, keeping the extremes; the journal keeps every sample.
An optimization view says **Force criterion met**, **Step budget reached; force
criterion not met** or **Step budget finished (no force target)**.

### Staleness

The state shown is the last one the run recorded, not a check on the job or the
process. A run with no final event is marked **Stale?** when no update has come
for more than 10 minutes or 20 times its median sample interval, whichever is
longer; this is a warning, not proof that the job died. A viewer still reading a
long journal says so rather than showing an early state as current. A run
killed by `SIGKILL` can stay **Running** in its journal, so do not use the
viewer as the only check of a job's health.

### JSON snapshot

`so3lr show PATH --json` prints one snapshot for scripts and agents:

```json
{
  "schema": 1,
  "runs": [
    {
      "journal": "/path/to/so3lr-run/run.<run-id>.events.jsonl",
      "run_id": "<run-id>",
      "state": "Running",
      "recorded_state": "Running",
      "age_seconds": 12.4,
      "operation": "md",
      "model": "so3lr-tauri-m-2606",
      "engine": "flash",
      "atoms": 192,
      "precision": "mixed",
      "device_count": 1,
      "ensemble": "nvt",
      "output_file": "trajectory.extxyz",
      "host": null,
      "slurm_job": null,
      "done": 45000,
      "planned": 100000,
      "fraction_done": 0.45,
      "last_sample": {"step": 45000, "time_ps": 22.5, "temperature_K": 301.2,
                      "potential_energy_eV": -2963.1, "energy_eV": -2938.4,
                      "volume_A3": null, "seconds_per_step": 0.0021},
      "checkpoint": {"step": 40000, "path": "restart.npz"},
      "message": null,
      "failure_reason": null
    }
  ]
}
```

`state` is the recorded state qualified by the viewer (`Stale?`, `Reading`,
`Read error`); `recorded_state` is what the journal says (`Started`, `Running`,
`Completed`, `Failed`, `Interrupted`). `done` and `planned` count MD steps of
the current invocation, optimization steps (`operation` `optimization`), or
finished training epochs (`training`). `host` and `slurm_job` are filled when the
run recorded them. Fields that the journal does not record are `null`.

## Batch output

With plain output (a redirected stderr, as in a Slurm `.out` file), MD adds one
progress line at every checkpoint, and at least every ten minutes:

```text
Progress 45.0% · step 450,000/1,000,000 · 2.31 ns/day · ETA 1d 02:13:00
```

Throughput and ETA use the wall time since the first MD sample, including
output. A failed or interrupted MD run ends with its last checkpoint and how to
continue from it:

```text
Last checkpoint: step 450,000 · restart.npz
Resume: --restart-load restart.npz (YAML: restart_load_path: restart.npz); relaxation is skipped and the trajectory is appended.
To finish the original length, also set --md-cycles 5500 (YAML: md_cycles: 5500); md_cycles counts from the checkpoint.
```

For a settings-file run, `so3lr resume` does the same in one step. A run with no
`restart_save_path` has no checkpoint, and the message says how to make runs
resumable. A `SIGTERM` or `SIGHUP` (`scancel`, a time limit) is recorded in the
journal and prints this hint, in batch jobs too; the process then still ends by
that signal, and a second signal ends it at once. Ctrl+C shows as
`Interrupted (Ctrl+C)`.
