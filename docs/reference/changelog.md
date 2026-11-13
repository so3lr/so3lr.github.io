# Changelog

This page lists model releases and notable package changes. The Python package
uses semantic versioning, and model names follow `so3lr-tauri-<size>-<YYMM>` and
never change (see [Models](../models.md#names-and-versions)). The short names
`so3lr-tauri-s` / `-m` / `-l` point to the newest release. To keep results
reproducible, use the dated name, e.g. `so3lr-tauri-m-2606`.

## Models

### 2606 — current default

Bundled as `so3lr-tauri-s-2606`, `so3lr-tauri-m-2606`, `so3lr-tauri-l-2606`. The
short names and the no-argument default resolve here.

Each 2606 checkpoint is the matching 2602 checkpoint refined further, with the
same architecture. The `fine_tuning.json` file in each model directory names
the parent run.

<!-- TODO: what changed relative to 2602 — training data, accuracy. See
     "Needed from you" in the site README. -->

### 2602

Bundled as `so3lr-tauri-s-2602`, `so3lr-tauri-m-2602`, `so3lr-tauri-l-2602`. First
release of the s/m/l family.

Relative to the original model, this generation:

- is trained on energies (the original was not)
- predicts polarizability-volume ratios (`a0_ratios`) instead of Hirshfeld
  volume ratios, and adds a dedicated `c6_ratios` head
- uses parameter-free NLH short-range repulsion instead of ZBL
- covers Z = 1–85 instead of eight biomolecular elements

### Original — `so3lr`

The model published in [JACS 147, 33723 (2025)](https://doi.org/10.1021/jacs.5c09558).
It is kept as `so3lr` to reproduce published results. It is a separate model
from the s/m/l family (see [Legacy SO3LR model](../models.md#legacy-so3lr-model)).

## Package

### Unreleased — on `main`

#### Added

- [`so3lr start`](../usage/cli.md#guided-setup) for guided MD setup with arrow-key structure and
  model menus, explicit review, and a portable structure/YAML/shell-script
  bundle. The first review action saves the bundle and starts MD locally;
  saving for later remains available.
- Live terminal MD plots and optimization status, plus offline
  [`so3lr show`](../usage/cli.md#offline-monitoring) views for individual runs
  and folders. Plain output is automatic when stderr is redirected; direct
  commands, YAML and shell/scheduler launchers remain supported.
- Per-invocation JSONL journals and resolved-settings YAML beside MD and
  optimization logs by default, including in plain and quiet modes. Text logs keep their
  numeric MD tables and add run boundaries, artifact paths and final summaries.
- `SO3LR_JOURNAL=0` disables both per-invocation sidecars while retaining text
  logs, trajectories, checkpoints and in-memory attached plots. The offline
  viewer requires an enabled journal.
- YAML `operation: optimization` selects the optimization command with its
  normal option validation. `so3lr --version` prints the installed package version.
- The terminal interface adds the required dependency `rich>=14,<16`; existing
  pinned environments must allow a compatible Rich version.
- Saved-run views show the age of the latest journal update, recorded host and
  Slurm identity, and qualified stale/history-loading states without querying
  or signalling the scheduler or producer process.
- Esc closes saved-run views and the folder overview, or hides attached MD
  plots without stopping the calculation. `q` retains its existing close/toggle
  behavior. In `so3lr start` menus, Esc or Backspace goes back one step.
- Guided setup asks only the essentials, then **Edit settings…** on the review
  changes one group at a time. It asks for a trajectory frame interval in ps,
  reads charge and multiplicity from the XYZ comment line, and explains
  impossible electron counts. Three example structures ship with the package.
- One terminal palette across the live view, `so3lr show` and setup (amber
  accent, cyan energy, violet volume), with taller plots and a progress bar in
  the live header and in wide run overviews.
- `so3lr start` first asks for a task: molecular dynamics, geometry
  optimization, dataset evaluation, watching runs, or the commands for
  fine-tuning, training and k-space tuning.
- A once-a-day, opt-out release notice in interactive terminals
  (`SO3LR_NO_UPDATE_CHECK=1`).
- Plain and batch output logs a `Progress … · ns/day · ETA …` line at every
  checkpoint and names the `so3lr show` command for the run. Failed or
  interrupted MD runs end with the last checkpoint and the option to resume
  from it; runs stopped by Ctrl+C or a signal show as **Interrupted**.

- `--precision mixed`, now the default: a float32 model around a float64 MD
  state (positions, momenta, box, energy sums). `--precision` takes
  `float32`, `mixed` or `float64` for MD, `opt`, `eval` and `tune-ewald`, on both
  engines, and as `precision:` in settings files. Every MD and `opt` run logs
  its precision with the model's and the MD state's dtypes.
- `precision=` on `So3lrCalculator` and `FlashCalculator`, with the same three
  values. Unset, it is `mixed` with `jax_enable_x64` and `float32` without.
  `dtype=` keeps working as the model's dtype.
- `so3lr.flash.domains.FlashDomains`: flash energies and forces for systems
  larger than one GPU holds, a domain at a time, with the same result as one
  call. Real-space long range only; see the
  [API reference](api.md#flashdomains).
- **The flash MD engine**: GPU kernels (JAX Pallas) that evaluate
  `so3lr-tauri-s/m/l` with no per-edge arrays. They give the same energy and
  forces as the neighbour-list potential in a few KiB per atom. `so3lr nve`, `nvt`,
  `nvt-langevin` and `npt` use it by default (`--engine auto`) wherever it
  applies (real-space long range or PME, isotropic NPT), and the log says
  which engine runs and why. `--engine standard` keeps the neighbour-list
  potential, and `python -m so3lr.flash.verify` compares the two. See
  [Molecular dynamics](../usage/md.md).
- `--flash-devices N` (`flash_devices`): flash on N GPUs of one node. Each GPU
  computes an equal share of the atoms, and energies and forces match those on
  one GPU.
  NVT water with PME on 8 H200s runs 6.2× (98,304 atoms) and 6.7× (192,000
  atoms) faster than on one.
- Locked environments managed by uv: a committed `uv.lock`, Git
  dependencies at immutable commits, and Python 3.12 and 3.13 coverage,
  including a smoke test of the built wheel. See [Installation](../installation.md).
- SHA-256 checksums for every bundled model file in `so3lr/models/SHA256SUMS`;
  see [Verifying the bundled files](../models.md#verifying-the-bundled-files).
- The Apache-2.0 licence text and a third-party notice for the vendored Google
  Research robust loss.
- Enhanced sampling on the flash engine: with `--engine auto` on a GPU,
  PySAGES runs on flash in NVT or NVE, on one GPU, in a periodic box. See
  [Enhanced sampling](../advanced/enhanced-sampling.md#engines).
- CUDA kernels for flash's `fast` memory mode on H100/H200 (compute capability
  9.0), built with `nvcc` on first use: a `so3lr-tauri-m` force call is 3.3×
  faster than on the Pallas kernels. `--flash-kernels` / `flash_kernels` /
  `FlashCalculator(flash_kernels=)` chooses `auto`, `pallas` or `cuda`. See
  [CUDA kernels](../usage/md.md#cuda-kernels).
- Semi-isotropic NPT on the flash engine.
- The proton-transfer CVs `chargeseparation` and `ionsitedistance`, and
  `r0_multiplier` for `coordinationnumber` (default 1.6, the value used
  before). PySAGES is pinned to a newer commit of the same fork.
- A cylindrical restraint (`cylinder_restraint_*`) on a group's centre of mass,
  for host–guest binding with enhanced sampling.
- A PySAGES restart warns about every `pysages_input` setting that differs from
  the method in the `.pkl`, which the run keeps.

#### Changed

- Bare `so3lr` opens guided setup in an interactive terminal outside recognized
  scheduler jobs; otherwise it prints help. Explicit simulation commands and
  `--settings` do not open the wizard. Help is grouped by purpose; `--help-full`
  shows the overview with additional usage examples.
- The former every-option `--help-full` output is replaced by that overview;
  use `so3lr COMMAND --help` for a command's complete options and defaults.
- Supplying command options without required input, malformed MD YAML, and
  combining root `--settings` with a subcommand now produce usage errors (exit
  two). Bare subcommands still show help (exit zero), runtime failures remain
  nonzero, and optimization execution success does not imply force convergence.
- `show` uses stable path ordering. `show --once` is a human snapshot without
  interactive hints or paging controls; its successful exit means rendering
  succeeded, independently of the recorded run state. Explicit non-journal
  file inputs are rejected with guidance to the journal directory.
- Redirected `--display rich` falls back to plain output, avoiding terminal
  control frames in batch output. Verified failures identify the absolute
  traceback log path and keep multiline detail in the log.
- `so3lr start` is the visible guided-setup command; `so3lr init` remains a hidden
  compatibility alias. Global display/verbosity/color options go before the
  subcommand. Plain and quiet modes keep per-invocation journal and resolved-YAML
  sidecars unless explicitly disabled with `SO3LR_JOURNAL=0`.
- Optimization's completion message is now `Optimization execution finished.`
  The summary distinguishes reaching the force criterion from exhausting the
  step budget; successful command completion alone does not establish convergence.
- Help no longer prints the ASCII banner; additional examples use `--help-full`.
  The decorative separator before MD completion is removed, and force-criterion
  diagnostics use significant digits. Numeric MD table columns are preserved.

- `--flash-mixed` / `flash_mixed` is deprecated in favour of `--precision`:
  on means `mixed` and off `float32`. It still works, with a warning.
- An explicit `--precision float32` now runs float32 throughout. It
  previously kept a float64 MD state unless `--no-flash-mixed` was given.
  The default run is unchanged.
- `tune-ewald` builds positions and the box in the MD state's dtype, float64
  by default, as MD does.
- `So3lrCalculator` and `FlashCalculator` raise an error for an explicit
  `mixed` or `float64` without `jax_enable_x64`, before the model loads.
- `so3lr tune-ewald`'s error estimates and search are SO3LR's own implementation
  of the Kolafa–Perram (Ewald) and Petersen (PME) estimates, with the same
  results. No torch-pme or JAX MD code remains, so the BSD-3-Clause licence is
  no longer needed. In `so3lr.cli.tune_ewald`, the error-bound objects are called
  with a candidate's parameters (no `.error()`), and `GridSearchTuner`,
  `TuningTimings`, `TuningErrorBounds` and the `exponent` arguments are gone.
- The s/m/l models are the SO3LR Tauri family: `so3lr-tauri-{s,m,l}-2606`
  and `-2602`, with floating aliases `so3lr-tauri-s/m/l`. The default is
  `so3lr-tauri-m-2606`, and the original model stays `so3lr`. The old names
  (`so3lr-m-2606`, `so3lr-m`, ...) and the `so3lr-o` synonym are removed.
- flash: the attention kernels' launch follows the model's filter width, and
  the backward pass's filter matmuls run as TF32×3 on tensor cores. The
  `so3lr-tauri-l` force call is 1.9× faster, `so3lr-tauri-s/m` about 10%; float32 errors and
  float64 results are unchanged.
- The JAX stack is capped (`jax<0.11`, `jaxlib<0.11`, `flax<0.12.9`), so a
  compatible existing cluster install is kept. Flax has a floor of 0.12.1:
  older releases fail to import JAX MD. Python is limited to 3.12 and 3.13.
- Weights & Biases is opt-in and installed through the `wandb` extra, not the
  base environment.
- TFDS-backed training uses finite, unit-consistent epochs and checkpoints the
  full optimizer, EMA and control state. A restart continues with the next
  epoch instead of replaying consumed batches. Params-only checkpoints from
  older versions continue from their weights with a fresh optimizer.
- Semi-isotropic NPT is marked experimental (a warning is logged once per run);
  fully anisotropic coupling remains unsupported.
- Documentation and the Colab notebook use the immutable `so3lr-tauri-{s,m,l}-2606`
  names and describe the pinned no-argument default, `so3lr-tauri-m-2606`.

#### Changed results and formats

- PME/P3M on non-orthogonal cells now passes the lattice vectors as rows, as
  jaxpme expects. Results on triclinic cells change (0.03–0.09 eV/Å in forces on
  triclinic MOFs). `Graph.cell` must hold the lattice as columns (`atoms.cell.T`),
  as glp and JAX-MD already do.
- The ASE calculator uses explicit periodic images when the minimum image is
  wrong: non-orthorhombic cells, or cell heights within twice the cutoff.
  Install the `matscipy` extra for a fast neighbour search.
- `so3lr eval` energies and forces change on datasets that carry `c6_ratios`
  labels: dispersion now uses the model's predicted C6, as the ASE calculator
  and MD always did.
- TFDS `filter.min_distance` and the absolute `filter.max_force` are applied;
  they were previously ignored or signed, so the selected training data changes.
  The `config.yaml` default `min_distance` is 0.5 Å.
- MD HDF5 trajectories are frame-major, `(n_frames, ...)`, instead of
  `(n_writes, save_buffer, ...)`; semi-isotropic boxes are stored as
  `(n_frames, 3, 3)`.
- MD `dipole_vec` without k-space is the molecular dipole; it previously
  reported the total charge. A `pbc=False` input that carries a cell runs in
  vacuum instead of periodically.
- Pre-MD relaxation frames go to `<output>_opt.xyz` instead of the start of the
  MD trajectory.
- On a GPU, enhanced-sampling runs move to the flash engine under
  `--engine auto`. Their trajectories match the neighbour-list engine's only up
  to rounding; `engine: standard` reproduces runs of earlier versions.
- The same holds for semi-isotropic NPT, and on an H100/H200 for flash runs that
  move to the CUDA kernels (`--flash-kernels pallas` keeps the Pallas ones).

#### Fixed

- Guided setup validates supported periodic cells, reserves the backend's 3%
  initial NPT contraction margin, and couples the small-cell 2 Å electrostatic
  smearing to an explicit 1 Å PME mesh spacing. Dispersion keeps its 2 Å
  smearing and 3 Å mesh spacing. These remain starting proposals rather than
  convergence guarantees; direct flags and existing YAML defaults are unchanged.
- Setup exports create files exclusively, recover from malformed XYZ elements,
  and use the CLI's shared save-buffer default. Logs and journals use consistent
  float32 energy reporting, and startup warnings target the current invocation.
- Terminal output supports ASCII fallback, restores state after interrupted
  initialization, and keeps controls visible in short optimization views.
  Saved-run readers tolerate malformed or excessively nested JSON records.
- MD/optimization errors retain the actual exception type and cause chain;
  their preambles are covered too. Errors are shortened only when the traceback
  was saved in the current log; logger setup/write failures and unexpected errors
  in other commands retain the full terminal traceback.
- Default `SIGTERM`/`SIGHUP` cleanup applies when the UI owns a terminal, including
  setup and `show`; noninteractive runs retain native termination. Defaults are
  restored when the first handler runs, before cleanup, so a second signal can terminate immediately. Custom
  signal handlers are unchanged; `SIGKILL` cannot be handled.
- SO3LR worker warnings reach the journal and live console without waiting for
  an MD reporting boundary when journaling is enabled. Scoped live-display
  redirection also keeps Python output to the current `sys.stderr` above the
  display. Third-party prebound streams and native stderr writes remain outside
  that routing, and their diagnostics are not automatically journaled.
- Hardware-memory reporting preserves termination exceptions instead of
  swallowing them. Verified traceback preservation also recognizes normal
  rotating-log rollover; unverifiable logging still keeps the full terminal trace.

- Semi-isotropic NPT on the neighbour-list engine drew new velocities at the
  start instead of taking the input structure's momenta.
- `so3lr tune-ewald --kspace-electrostatics pme` searches with 4 interpolation
  nodes by default. The previous default of 3 failed in jax-pme, whose PME
  only takes 4.
  P3M also defaults to 4 (it was 3) and takes 3–5. Other orders, and `nodes_lo`
  above `nodes_hi`, are refused before the model is built.
- The flash engine refuses elements the model was not trained on (Z > 85 for
  `so3lr-tauri-s/m/l`), as the neighbour-list engine does. Previously it ran
  them with reference values gathered from another element's row.
- `restart_pysages_load_path` without `restart_load_path`, or with a restart
  into another ensemble, is now an error up front. It previously crashed after
  setup.
- A second enhanced-sampling run in the same Python process reused the first
  run's compiled MD step. Runs from the command line, one per process, were
  unaffected.
- A PySAGES cycle repeated after a neighbour-list overflow left the CV values
  of the discarded attempt in `cv_vs_timestep.txt`, which then had more rows
  than steps. They now go to `cv_vs_timestep_discarded.txt`.
- `--engine standard` writes an `Engine:` line to the log, as auto and flash do.
- HDF5 trajectories store `a0_ratios` and `dispersion_energy_kspace` per atom.
  They were previously skipped with a warning.
- `So3lrCalculator(kspace_electrostatics=...)` selected the k-space *grid* but
  not the *solver*, which defaulted to PME, so `ewald` and `p3m` through the ASE
  calculator returned wrong electrostatic energies without erroring. `pme` and
  every CLI run were unaffected; regenerate any `ewald`/`p3m` numbers from the
  ASE calculator made before the fix.
- Fine-tuning strategies that name a `layers_*` entry (`last_layer`, `ll_fmlp`,
  `ll_obs_023`, `last_layer_and_final_mlp`, `first_layer_and_last_layer`) took
  the layer count from the *new* config instead of the checkpoint being refined,
  so they froze or unfroze the wrong layer whenever the two disagreed. `full`
  was unaffected.
- The Ewald tuner's reciprocal-space error estimate uses the Kolafa–Perram
  (LAMMPS) form, which matches the measured force error; the previous estimate
  underestimated it 4–36×.
- The ASE calculator no longer minimum-images a `pbc=False` molecule that
  carries a bounding cell.
- Periodic calculations reject unsupported mixed-axis periodicity, including in
  mixed batches, and dispersion PME uses a residual cutoff consistent with the
  configured damping split. Padded PME/Ewald batches have finite derivatives
  with respect to predicted C6.
- Evaluation keeps reference C6 labels separate from model predictions, and
  `--overwrite` replaces old predictions and embeddings instead of appending.
  Three-atom molecular dipoles are preserved, atom selections apply to HDF5
  embeddings, and requested embeddings reach EXTXYZ output.
- MD output keeps partially filled write buffers, writes frames before the
  restart checkpoint, protects existing HDF5 trajectories on restart, and moves
  each output into place only once it is complete. The NPZ restart keeps the
  previous checkpoint if serialization fails, and is loaded with
  `allow_pickle=False`.
- Dimer calculations check the selected model's supported elements before
  evaluating.
- A rejected training restart leaves the checkpoint metadata untouched, and
  ASE/NPZ fine-tuning resumes weights and step from its own target checkpoint
  instead of silently restarting from the parent model.
- Configuration unit expressions are parsed by a constrained unit parser
  instead of Python `eval`.
- Example notebooks: imports updated to the installed `so3lr` package;
  thermodynamic units corrected (NVT converts its target temperature to energy
  units; the NPT examples convert pressure and give the box three lengths); the
  production notebook flushes its final HDF5 buffer; every MD notebook fails
  loudly on neighbour-list overflow. The Colab notebook pins its uv bootstrap,
  rejects partially periodic NPT inputs and stops when a step fails.

<!-- TODO: backfill earlier package releases from the git history / tags. -->

## Reporting a regression

If results change across releases in a way this page does not explain, please
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues)
with the model names and the top of your log file. The log records the resolved
alias, so it shows which checkpoint produced the numbers.
