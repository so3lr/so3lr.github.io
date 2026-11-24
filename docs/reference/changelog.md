# Changelog

This page lists model releases and notable package changes. The Python package
uses semantic versioning, and model names follow `so3lr-tauri-<size>-<YYMM>` and
never change (see [Models](../models.md#names-and-versions)). The short names
`so3lr-tauri-s` / `-m` / `-l` point to the newest release. To keep results
reproducible, use the dated name, e.g. `so3lr-tauri-m-2606`.

## Models

### 2606 { #model-2606 }

The current default. Bundled as `so3lr-tauri-s-2606`, `so3lr-tauri-m-2606`, `so3lr-tauri-l-2606`. The
short names and the no-argument default resolve here.

Each 2606 checkpoint is the matching 2602 checkpoint refined further, with the
same architecture. The `fine_tuning.json` file in each model directory names
the parent run.

<!-- TODO: what changed relative to 2602 — training data, accuracy. See
     "Needed from you" in the site README. -->

### 2602 { #model-2602 }

Bundled as `so3lr-tauri-s-2602`, `so3lr-tauri-m-2602`, `so3lr-tauri-l-2602`. First
release of the s/m/l family.

Relative to the original model, this generation:

- is trained on energies (the original was not)
- predicts polarizability-volume ratios (`a0_ratios`) instead of Hirshfeld
  volume ratios, and adds a dedicated `c6_ratios` head
- uses parameter-free NLH short-range repulsion instead of ZBL
- covers Z = 1–85 instead of eight biomolecular elements

### Original — `so3lr` { #model-original }

The model published in [JACS 147, 33723 (2025)](https://doi.org/10.1021/jacs.5c09558).
It is kept as `so3lr` to reproduce published results. It is a separate model
from the s/m/l family (see [Legacy SO3LR model](../models.md#legacy-so3lr-model)).

## Package

### 0.2.0 — 2026-11-24 { #v0-2-0 }

The [v0.2.0 release](https://github.com/general-molecular-simulations/so3lr/releases/tag/v0.2.0),
with the SO3LR Tauri models.

#### Added

- [`so3lr start`](../usage/terminal-ui.md#guided-setup) for guided MD setup with arrow-key structure and
  model menus, explicit review, and a portable structure/YAML/shell-script
  bundle. The first review action saves the bundle and starts MD locally;
  saving for later remains available.
- Live terminal MD plots and optimization status, plus offline
  [`so3lr show`](../usage/terminal-ui.md#so3lr-show) views for individual runs
  and folders. Plain output is automatic when stderr is redirected.
- Per-invocation JSONL journals beside MD, optimization and training logs by default,
  including in plain and quiet modes. Text logs keep their numeric MD tables.
- `SO3LR_JOURNAL=0` disables the journal while retaining text logs,
  trajectories, checkpoints and in-memory attached plots. The offline viewer
  requires an enabled journal.
- `so3lr --version` prints the installed package version.
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
  (`SO3LR_NO_UPDATE_CHECK` set to anything but empty or `0` turns it off; it
  never runs in scheduler or CI jobs).
- `eval`, `tune-ewald`, `finetune` and `train` use the live display too, with
  a progress bar, ETA and result tables, instead of falling back to log lines.
  `eval` reports binding energies for dimer files without `--save-to`.
- Optimization graphs: potential energy and log-scale maximum force with the
  force target, live and in `so3lr show`.
- Training graphs: validation loss and force MAE per validation, live and in
  `so3lr show` (`TRN` in the folder overview).
- Every run ends with one closing line: ✓ met its goal, `!` finished without it,
  ■ interrupted, ✕ failed. The activity line shows an ETA once a rate is measured.
- Log files add `◆ SO3LR TAURI · <version> · <date time>` under the logo.
- Plain and batch output logs a `Progress … · ns/day · ETA …` line at every
  checkpoint and names the `so3lr show` command for the run. Failed or
  interrupted MD runs end with the last checkpoint and the option to resume
  from it; runs stopped by Ctrl+C or a signal show as **Interrupted**.
- `so3lr tune-ewald` tunes PME on the flash engine whenever flash takes the
  structure. It takes its charges and timings from flash, so tens of thousands
  of atoms fit on one GPU: a 32,512-atom POPC bilayer tunes in about 3.5 min on
  an A100 with `so3lr-tauri-m` or `-l`, where the neighbour-list path ran out of
  memory. `--engine` / `engine:` (`auto`, `flash`, `standard`) picks the engine,
  as in MD. Ewald, P3M and the `cutoff` modes still tune on the neighbour lists.
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
- CUDA kernels for flash on NVIDIA GPUs, built with `nvcc` on first use: on a
  GPU of compute capability 9.0, a `so3lr-tauri-m` force call is 3.3× faster
  than on the Pallas kernels. `--flash-kernels` / `flash_kernels` /
  `FlashCalculator(flash_kernels=)` chooses `auto`, `pallas` or `cuda`. See
  [CUDA kernels](../usage/md.md#cuda-kernels).
- Semi-isotropic NPT on the flash engine.
- Small periodic boxes on the flash engine, down to about 1 Å across. Before,
  flash needed about 30 Å at the default 12 Å `lr_cutoff`, and a smaller box
  fell back to the neighbour-list engine, which still refuses boxes under twice
  `lr_cutoff` (24 Å). Flash sums the long range over the periodic images, and
  under twice the 4.5 Å short-range cutoff the short range too, in the Pallas
  and the CUDA kernels. On one A100 with `so3lr-tauri-m`, 64 waters in a
  12.4 Å box run at 1.5 ms per step with the CUDA kernels (2.0 ms with
  Pallas), against 15 ms for the neighbour-list engine on the smallest
  supercell it accepts, and 620 ms for `So3lrCalculator`. See
  [Box size](../usage/md.md#box-size).
- The proton-transfer CVs `chargeseparation` and `ionsitedistance`, and
  `r0_multiplier` for `coordinationnumber` (default 1.6, the value used
  before). PySAGES is pinned to a newer commit of the same fork.
- A cylindrical restraint (`cylinder_restraint_*`) on a group's centre of mass,
  for host–guest binding with enhanced sampling.
- A PySAGES restart warns about every `pysages_input` setting that differs from
  the method in the `.pkl`, which the run keeps.
- [`so3lr resume [PATH] [--or-start]`](../usage/cli.md#resuming-a-run)
  continues a settings-file MD run (a run folder or a settings file) from its
  checkpoint for the rest of its `md_cycles`, without relaxation. It appends to
  an EXTXYZ trajectory, writes HDF5 to a new `<stem>.from_cycle_N.h5` (or `.hdf5`), and
  carries the PySAGES bias, refusing to start when its file is missing.
  `--or-start` starts the run when there is no checkpoint yet.
- `so3lr start --input STRUCTURE` writes the guided MD setup without prompts,
  for scripts and agents; `--print` prints its `settings.yaml` instead. See
  [Without prompts](../usage/terminal-ui.md#without-prompts).
- `so3lr show --json` prints one machine-readable snapshot per run: state,
  progress, last sample, checkpoint and failure. See
  [JSON snapshot](../usage/terminal-ui.md#json-snapshot).
- `--kspace-dispersion`, `--kspace-dispersion-smearing`,
  `--kspace-dispersion-spacing` and `--kspace-dispersion-interp-nodes` on the MD
  commands and `opt`, and `auto` and `none` for `--kspace-electrostatics` and
  `--kspace-dispersion`.
- A run folder's `run.sh` runs `so3lr resume --or-start .`: it starts the run,
  or continues it from its checkpoint, from the folder it is in, and works under
  `sbatch run.sh`.
- `So3lrEngine`: energy and forces of a fixed set of atoms for new positions
  and a new box at each call, for MD codes outside JAX-MD (OpenMM, i-PI,
  LAMMPS, your own loop). For each box it runs on flash, the neighbour lists
  or exact periodic images, whichever is the first that can, and it takes
  k-space electrostatics (`ewald`, `pme`, `p3m`) and k-space dispersion
  (`pme`, `ewald`). See the [API reference](api.md#so3lrengine).
- `kspace_remove_net_force` (default `true`): MD and `opt` subtract the mean
  force whenever a k-space term is on. See
  [Net force](../advanced/electrostatics.md#net-force).
- Three fine-tuning recipes that keep the long-range terms intact on data with
  energies and forces only: `training.long_range_anchor_weights` anchors the
  charges, a0/C6 ratios and optionally dipoles to the starting model;
  `--strategy readout_only` with a residual energy readout
  (`model.energy_residual_mlp_widths`, `energy_residual_inputs`) keeps them
  exactly; and a per-atom `partial_charges` target, read from an extxyz array,
  trains on a teacher's predictions. See
  [Choosing a strategy](../training/finetuning.md#choosing-a-strategy).
- A fine-tune config may set `model.cutoff_lr` and the k-space keys
  (`kspace_electrostatics`, `kspace_smearing`, `kspace_spacing`,
  `kspace_dispersion` and its smearing and spacing) for training on periodic
  data. They are recorded in `fine_tuning.json` and are not saved with the
  model. See [Periodic data](../training/finetuning.md#periodic-data).
- `model.kspace_dispersion: false` keeps dispersion in real space when training
  with k-space.
- `training.data_parallel`: training on all local GPUs with a TFDS dataset,
  one batch per device per optimizer step. See
  [Multiple GPUs](../training/from-scratch.md#multiple-gpus).
- Flash's CUDA kernels for systems of millions of atoms: where every layer's
  per-pair data does not fit in memory, they make it a chunk of rows at a time,
  in the `fast` and `default` memory modes, and `--flash-mode auto` takes the
  fastest that fits. MD of water with PME ran on them up to 16 million atoms
  with `so3lr-tauri-s/m` and 9 million with `-l` on one large-memory GPU, at
  2.2–2.8× the
  speed of the Pallas kernels.
- Flash's CUDA kernels run on any NVIDIA GPU of compute capability 8.0 or
  newer whose shared memory per block holds the model's tiles, which are sized
  to fit. At 163 KiB per block, a force call is 1.65–2.1× faster than on the
  Pallas kernels; at 99 KiB, `so3lr-tauri-l` stays on the Pallas kernels.
- `--flash-mode auto` logs why a run is on the Pallas kernels
  (`Flash: CUDA kernels not used (...)`), as a memory mode given did.
- The flash engine holds `fixed_atom_indices` in `nve`, `nvt` and
  `nvt-langevin`, for example a solute held in a periodic water box. NPT,
  `mobile_atom_indices` and `atom_mask_path` run on the neighbour-list engine.

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
  traceback log path.
- `so3lr start` is the visible guided-setup command; `so3lr init` remains a hidden
  compatibility alias. Plain and quiet modes keep the per-invocation journal
  unless explicitly disabled with `SO3LR_JOURNAL=0`.
- Optimization's completion message is now `Optimization execution finished.`
  The closing line distinguishes reaching the force criterion from exhausting the
  step budget; successful command completion alone does not establish convergence.
- `so3lr --help` opens with a runnable 1 ps NVT example and groups the
  commands by task; every help screen ends with a link to this site.
  `so3lr --version` prints `SO3LR v<version>`.
- Help no longer prints the ASCII banner; additional examples use `--help-full`.
  The decorative separator before MD completion is removed, and force-criterion
  diagnostics use significant digits. Numeric MD table columns are preserved.
- `tune-ewald`'s `mesh_lo` / `mesh_hi` now mean 2ⁿ mesh points along the shortest
  cell vector, the mesh jax-pme builds. They used to give the spacing
  2*L*/(2ⁿ − 1), which jax-pme padded to 2ⁿ⁺¹ points. The defaults moved from
  4–6 to 5–7, which tries the same meshes. A settings file that sets these keys
  now gets meshes half as fine; the log lists the meshes tried.
- `tune-ewald`'s PME error estimate uses the mesh spacing jax-pme actually
  builds, after padding the mesh to a power of two. The estimates are about 4–9×
  lower than before, so coarser meshes now reach a given `accuracy`.
- `tune-ewald` measures the real-space error on the pairs of up to 2048 sampled
  atoms, instead of on every neighbour pair.
- `tune-ewald` recommends the most accurate candidate within `accuracy` whose
  timing is within 3 % of the fastest, not simply the fastest.
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
- `--display`, `--verbose`/`-v`, `--quiet`/`-q` and `--no-color` may also
  follow the subcommand (`so3lr nvt --input x.xyz -q`).
- Unknown settings-file keys are dropped after their warning. Kept, they made
  `--engine auto` fall back to the neighbour-list engine.
- A slab or wire (mixed periodicity) is a usage error (exit 2) named before
  anything starts.
- `tune-ewald`'s MD command and settings for the `cutoff` modes turn k-space off
  explicitly (`--kspace-electrostatics none --kspace-dispersion none`).
- On `SIGTERM` or `SIGHUP` (`scancel`, a time limit), batch runs too record the
  interruption in the journal and print the resume hint before the process ends
  by the signal.
- A failed run prints its whole error message, not only the first line.
- Live plots redraw at most once a second, text views every 0.25 s.
- An MD restart sets the centre-of-mass velocity to zero, and warns when the
  box was drifting, except with `kspace_remove_net_force: false`,
  `nvt-langevin`, fixed atoms or a restarted PySAGES state. See
  [Restarts](../usage/md.md#what-a-restart-preserves).
- `--strategy freeze_long_range` and its alias `forces_only` are removed. They
  froze the long-range heads, but the heads read features that keep training,
  so the charges and ratios still drifted. Asking for either names the
  [recipes](../training/finetuning.md#choosing-a-strategy) that keep the
  long-range terms.
- `so3lr train` and `so3lr finetune` refuse periodic ASE/NPZ data unless
  `model.cutoff_lr` is set and at most `data.neighbors_lr_cutoff`. Before, the
  long-range terms were cut off where the pair list ends.
- Training with k-space sizes the mesh from the longest cell row per axis over
  all training and validation structures (the first batch with TFDS), not from
  the first cell.
- Loading a training workdir without a `params.pkl` restores the checkpoint
  with the lowest validation loss, or the newest when none saved a loss. Before,
  it loaded the newest of the three best. Bundled models are unaffected.
- JAX 0.10.2 is excluded (`jax`, `jaxlib` and `jax[cuda12]`): on the CPU it
  miscompiles the periodic displacements of JAX MD and returns wrong energies
  and forces without an error. The lock moves to 0.10.1, and an existing
  0.10.2 install is moved to 0.10.1; install with the `cuda12` extra so the
  CUDA plugin moves too.
- JAX MD is pinned to SO3LR's fork rebased onto current upstream JAX MD,
  which adds `equinox` and `jaxtyping` to the dependencies.
- Flash sorts the atoms along a space-filling curve in every force call and
  returns the forces in the input order, so a structure whose atoms come in
  random order runs as fast as an ordered one.
- Locked dependencies are updated for current security advisories. The `tfds`
  extra moves to TensorFlow and TensorBoard 2.21 with a patched setuptools,
  and the lock moves h5py from 3.16 to 3.14.

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
- The same holds for semi-isotropic NPT, and on NVIDIA GPUs of compute
  capability 8.0 or newer for flash runs that move to the CUDA kernels
  (`--flash-kernels pallas` keeps the Pallas ones).
- For the SO3LR Tauri models, the command line (MD, `opt`, `eval`,
  `so3lr --settings`) uses PME for
  electrostatics and dispersion on structures periodic along all three axes,
  with a 10 Å cutoff, 2 Å smearing and a 1.55 Å mesh (dispersion 2 Å and 3 Å).
  A cell too small for 10 Å gets 0.45 × its smallest height and a 1 Å mesh
  (MD and `opt`). Before, k-space was off unless asked for and the cutoff was
  12 Å. `--kspace-electrostatics none --kspace-dispersion none` (or `none` in a
  settings file) restores real space. With k-space electrostatics asked for
  explicitly, an unset smearing and mesh are now 2 Å and 1.55 Å instead of 4 Å
  and 2 Å. Non-periodic runs, `tune-ewald` and the Python calculators are
  unchanged, and so is the original `so3lr` model (and models fine-tuned from
  it), which keeps real space at 12 Å unless a k-space method is named. See
  [Long-range defaults](../usage/cli.md#long-range-defaults).

#### Fixed

- Guided setup validates supported periodic cells and reserves the backend's
  3% initial NPT contraction margin.
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
- Default `SIGTERM`/`SIGHUP` cleanup restores the terminal in runs, setup and
  `show`. Defaults are restored when the first handler runs, before cleanup, so
  a second signal can terminate immediately. Custom signal handlers are
  unchanged; `SIGKILL` cannot be handled.
- SO3LR worker warnings reach the journal and live console without waiting for
  an MD reporting boundary when journaling is enabled. Scoped live-display
  redirection also keeps Python output to the current `sys.stderr` above the
  display. Third-party prebound streams and native stderr writes remain outside
  that routing, and their diagnostics are not automatically journaled.
- Hardware-memory reporting preserves termination exceptions instead of
  swallowing them. Verified traceback preservation also recognizes normal
  rotating-log rollover; unverifiable logging still keeps the full terminal trace.
- `tune-ewald`'s search ranges no longer go one step past their end when the
  range is not a whole number of steps (`cutoff_lo` 7 with `cutoff_hi` 12 and
  step 2 tried 6 Å as well).
- `so3lr tune-ewald` passes `--spin` to the model (the CLI dropped it).
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
- An existing output is refused before the structure is read or the run's
  journal opens, so a refused start leaves nothing behind.
- `md_dt` (`--dt`) must be positive, from the command line and in settings
  files.
- MD with k-space electrostatics or dispersion drifted as a whole: PME's mesh
  leaves a small net force, which built up into a centre-of-mass drift that
  inflated mean-squared displacements. MD and `opt` now remove it, on both
  engines; `kspace_remove_net_force: false` gives the previous forces.
- `so3lr opt` and the relaxation before MD ignored `kspace_dispersion` and ran
  real-space dispersion, while the MD that followed used k-space dispersion.
- `so3lr tune-ewald` estimated the real-space errors with fixed electrostatic
  and dispersion scales (4.0 and 1.2) instead of the model's. The 2606 models
  and `so3lr-tauri-m-2602` have a dispersion scale of 1.0, so their dispersion
  cutoff errors were estimated for another damping width.
- A `So3lrCalculator` that evaluated a structure in free space and then in a
  box reused the free-space neighbour list and returned the free-space energy
  and forces for the box.
- Fine-tuning dropped every `model:` key, so periodic data trained with the
  pretrained 1000 Å `cutoff_lr`, cut off where the pair list ends, and k-space
  could not be turned on.
- Training batches that mix molecules and crystals failed (crystal first) or
  silently dropped the crystals' long-range image offsets (molecule first).
- Training with k-space refuses structures that are not periodic in all three
  directions. They got only the screened real-space Coulomb, without an error.
- TFDS training saves the steps after the last evaluation. Before, a run that
  ended between evaluations lost them, and a resume started from the last
  periodic checkpoint.
- Training keeps the newest checkpoint as well as the three with the lowest
  validation loss. Before, a newest checkpoint outside the best three was
  deleted, and a restart went back to an older step without a warning.
- `so3lr eval --engine auto` falls back to the neighbour-list model for a
  structure with no box whose atoms are too spread out for flash's cell grid,
  such as the separated frame of a dimer file. It previously chose flash and
  failed during the evaluation. Every such structure is checked, not only the
  first, and one with no atoms or positions that are not finite is declined.
- NPT on the neighbour-list engine reallocates a cell list whose box has
  shrunk below its cells. The run previously went on with a list that missed
  pairs near the cutoff.

<!-- TODO: backfill earlier package releases from the git history / tags. -->

## Reporting a regression

If results change across releases in a way this page does not explain, please
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues)
with the model names and the top of your log file. The log records the resolved
alias, so it shows which checkpoint produced the numbers.
