# FAQ

## Using the model

??? question "Are absolute energies meaningful?"
    With the `so3lr-tauri-s`/`-m`/`-l` models, yes, because they are trained on
    energies. With the legacy `so3lr` model, only differences between
    structures of the same composition are meaningful. Units are eV and Å.

??? question "What should `lr_cutoff` be?"
    | System | Setting |
    |---|---|
    | Periodic / condensed phase | `12.0` (the default) |
    | Gas phase, isolated molecules | `100` |
    | Periodic and charged, or cutoff artifacts matter | `12.0` plus [k-space electrostatics](../advanced/electrostatics.md) |

    At `12.0`, electrostatics and dispersion of an isolated molecule are
    truncated. At `100`, the long-range neighbor list of a large periodic box
    becomes enormous.

??? question "Which model should I use?"
    The default, `so3lr-tauri-m`, is a good place to begin and is the model we
    use for biomolecular MD. `so3lr-tauri-l` is the most accurate and is
    recommended for reactions, but it is slower. Use `so3lr` only to reproduce
    the original JACS paper. See [Models](../models.md).

??? question "Which elements are supported?"
    The `so3lr-tauri-s/m/l` models cover H to At (Z = 1–85), and the original
    `so3lr` covers H, C, N, O, F, P, S and Cl. See
    [Models](../models.md#the-current-family).

??? question "Why did my results change after upgrading?"
    Usually this is because `so3lr-tauri-s` / `-m` / `-l` point to the newest
    release. Pin the dated name, e.g. `so3lr-tauri-m-2606`, wherever numbers
    must stay reproducible. If you already do, check the
    [changelog](changelog.md).

??? question "`float32` or `float64`?"
    `float32` is mixed precision: the model runs in float32, while MD keeps
    positions, momenta and energy sums in float64. It is roughly twice as fast
    as `float64` and is what we recommend for MD. Use
    `float64` for Hessians and tight force convergence. On the command line,
    pass `--precision float64`. In Python, enable JAX's 64-bit mode before
    importing SO3LR, otherwise `dtype=np.float64` has no effect (see the
    [API reference](api.md)).

??? question "Can I use a fine-tuned or custom model?"
    Yes. Pass the directory path anywhere a model name is accepted. It needs a
    `hyperparameters.json` and either a `params.pkl` or an orbax `checkpoints/`
    folder, which is what fine-tuning writes.

## Running simulations

??? question "What timestep should I use?"
    Use a timestep of `0.5` fs. SO3LR has no bond constraints, so do not use
    the 2 fs timestep of a classical force field.

??? question "My energy drifts."
    Try, in order, `dt = 0.25` fs, raising `--buffer-sr` / `--buffer-lr` above
    `1.25`, and `--precision float64`. A drift watchdog logs a warning when the
    conserved quantity drifts faster than `energy_drift_threshold` (default
    0.1 meV/atom/ps).

??? question "How do I restart a killed job?"
    Save a checkpoint with `--restart-save checkpoint.npz`, then resume with
    `--restart-load checkpoint.npz`. See
    [what a restart preserves](../usage/md.md#what-a-restart-preserves).

??? question "How do I freeze part of my system?"
    This is only possible in the settings file, with `fixed_atom_indices` or
    `mobile_atom_indices`. See [Freezing atoms](../usage/md.md#freezing-atoms).

??? question "Is MD running on the flash engine?"
    Check the `Engine:` line in the log. It names the engine and, when flash
    was not used, the reason. `--engine flash` forces flash and fails with the
    reason if it cannot run. See
    [Molecular dynamics](../usage/md.md).

??? question "Can one simulation use several GPUs?"
    On the flash engine, yes: `--flash-devices N` splits one simulation across
    N GPUs of a single node. See
    [Molecular dynamics](../usage/md.md).

??? question "My trajectory files are enormous."
    Use `output_atom_indices` to write only the atoms you need, write `.hdf5`
    instead of `.xyz`, and raise `md_steps` so that output happens less often.

??? question "Why won't it overwrite my old trajectory?"
    SO3LR refuses to truncate an existing output file. Rename the old file, or
    pass `--overwrite` (`overwrite: true`). A restart appends to an EXTXYZ
    trajectory, but an HDF5 trajectory needs a new path.

??? question "How do I run longer than the queue walltime?"
    Save checkpoints with `restart_save_path`, then submit dependent jobs that
    resume from them. An example is in
    [Running on a cluster](../usage/cli.md#running-on-a-cluster).

??? question "How do I tell which model actually produced a trajectory?"
    The log header records the model name as you passed it, so pass a dated
    name such as `so3lr-tauri-m-2606` rather than an alias. See [the log file](../usage/cli.md#the-log-file).

??? question "Does SO3LR do enhanced sampling?"
    Yes, via [PySAGES](../advanced/enhanced-sampling.md), including ABF,
    metadynamics and harmonic bias. Install it with
    `uv sync --locked --extra pysages`.

## Errors

??? failure "`KeyError: 'hirshfeld_ratios'`"
    This happens when you ask a Tauri model for a legacy key. The
    `so3lr-tauri-s/m/l` models emit `a0_ratios`, and only the legacy `so3lr`
    model emits `hirshfeld_ratios`. See
    [names across model generations](../advanced/observables.md#names-differ-between-model-generations).

??? failure "My settings file key was ignored"
    Check the job's stderr, not the log. Unknown keys give a warning with a
    suggestion, not an error:

    ```
    Unknown setting 'md_t' in settings.yaml; ignoring. Did you mean: md_dt, md_T, md_P?
    ```

??? failure "K-space electrostatics seems to have no effect"
    The ASE calculator parameter is `kspace_electrostatics`, and it requires
    `kspace_smearing` and `kspace_spacing`. A misspelled keyword is not
    rejected, so it can look accepted while doing nothing. K-space arguments are
    ignored for non-periodic systems. See [Long-range electrostatics](../advanced/electrostatics.md#from-the-ase-calculator).

??? failure "Neighbor list overflow / silently missing interactions"
    If you write your own JAX-MD loop, you need to check
    `nbrs.did_buffer_overflow` and reallocate at each step. The
    [CLI](../usage/cli.md) does this for you. See
    [Custom integrations](../advanced/integrations.md#driving-jax-md-directly).

??? failure "`Calculating stress and observables gradients at the same time is not supported`"
    `calculate_hessian`, `calculate_stress` and observable `_grad`/`_jac`
    outputs are mutually exclusive, so run them as separate calculations.

??? failure "MD refuses my periodic box as too small"
    Every box length must be at least twice `lr_cutoff` (24 Å at the default
    12 Å). Lower `--lr-cutoff` and let k-space carry the long range
    (`--kspace-electrostatics pme`, plus `kspace_dispersion: pme` in a settings
    file, or dispersion stays truncated), or use the
    [ASE calculator](../usage/ase.md). See [Box size](../usage/md.md#box-size).

??? failure "Installation fails on a cluster"
    Usually the Python version is outside 3.12–3.13, or JAX is newer than
    SO3LR supports. See
    [installation troubleshooting](../installation.md#troubleshooting).

## Fine-tuning

??? question "My dataset only has forces. Can I fine-tune?"
    Yes, but use `--strategy freeze_long_range`. With `--strategy full`, the
    forces improve while electrostatics and dispersion silently degrade. See
    [Fine-tuning](../training/finetuning.md#choosing-a-strategy).

??? question "How do I check a fine-tune didn't break anything?"
    Evaluate it on held-out data with `so3lr eval`, including dipole and ratio
    targets, and compare against the base model. See
    [Checking the result](../training/finetuning.md#checking-the-result).

## Getting help

If you are still stuck, or find errors or inconsistencies in the code or these
docs, please
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues).
Include the top of your log file.
