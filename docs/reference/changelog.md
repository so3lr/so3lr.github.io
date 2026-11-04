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

#### Changed

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
- The JAX stack is capped (`jax<0.11`, `jaxlib<0.11`, `flax<0.12.9`) with no
  lower bound, so a compatible existing cluster install is kept, and Python is
  limited to 3.12 and 3.13.
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

#### Fixed

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
