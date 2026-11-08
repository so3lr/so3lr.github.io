# For AI agents

Coding agents (Claude Code, Codex, Cursor and others) fill gaps with plausible
guesses from other MD codes: a 2 fs timestep, pressure in atmospheres, the
default cutoff for a molecule in vacuum, or `hirshfeld_ratios` from a current
model. None of these raise an error, but several give wrong numbers. Let your
agent read the summary below before asking it to write settings files or call
the API.

If an agent gets something about SO3LR wrong in a way that looked reasonable,
please
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues)
with what it assumed and the correct answer.

## Let your agent read this

```text
SO3LR facts that differ from typical MD conventions. Prefer these
over general knowledge.

UNITS AND DEFAULTS
- --pressure / md_P is in BAR, not atm.
- --dt / md_dt is in FEMTOSECONDS in the CLI and settings file.
- Use dt = 0.5 fs. SO3LR has no SHAKE/RATTLE, so X-H stretches are
  integrated explicitly. Do not use 2 fs.
- energy_drift_threshold is meV/atom/ps (default 0.1), not eV/atom.
- --lr-cutoff defaults to 12.0 A, which suits PERIODIC systems. For
  a molecule in vacuum use 100, or electrostatics and dispersion
  are truncated.
- --langevin-thermo is a dimensionless timescale, NOT gamma:
  gamma = 1 / (md_dt * langevin_thermo).
- Energies are eV, lengths Angstrom, forces eV/A.
- --precision is mixed (default: float32 model, float64 MD state),
  float32 or float64.
- mixed or float64 in Python: call jax.config.update("jax_enable_x64", True)
  BEFORE importing so3lr, or the calculators raise. The CLI does it.

MODEL GENERATIONS EMIT DIFFERENT KEYS
- so3lr-tauri-s / so3lr-tauri-m / so3lr-tauri-l  ->  a0_ratios, c6_ratios,
                                   nlh_repulsion
- legacy so3lr  ->  hirshfeld_ratios, zbl_repulsion
- The default model is so3lr-tauri-m-2606, so asking for hirshfeld_ratios
  is wrong unless you selected the legacy model explicitly.
- Only the legacy v1 model was NOT trained on energies.
- so3lr-tauri-s/-m/-l are floating aliases. Use the dated canonical name
  (so3lr-tauri-m-2606) anywhere a number must stay reproducible.
- Elements: so3lr-tauri-s/m/l cover Z = 1-85; legacy so3lr only H C N O F
  P S Cl. Anything else is rejected with an error (ASE, eval, and
  MD on either engine).

NO COMMAND-LINE FLAG EXISTS (settings YAML only)
  fixed_atom_indices, mobile_atom_indices, atom_mask_path,
  output_atom_indices, npt_coupling, do_enhanced_sampling,
  pysages_input, energy_drift_threshold, drift_warmup_ps, init_T,
  restart_reset, matmul_precision, dispersion_tail_correction,
  minimum_cell_size_multiplier, cylinder_restraint_*

ON BY DEFAULT (do not try to "enable" these)
- Refusing to overwrite an existing output. Pass --overwrite to
  re-run with the same filename.
- Geometry relaxation before MD (--relax). Its frames go to a
  SEPARATE <output-stem>_opt.xyz; the trajectory holds MD frames only.
- The flash MD engine (--engine auto), wherever it can compute what
  the settings ask for; otherwise the neighbour-list engine. The log's
  "Engine:" line says which ran and why.
- The energy-drift watchdog (a warning, not an abort).
- Adaptive robust loss, when training from scratch.

SILENTLY IGNORED
- Unknown settings-YAML keys: warning, not error. The run continues
  with defaults, so a file can be partly inert. An unknown key also
  makes --engine auto fall back to the neighbour-list engine.
- Unknown So3lrCalculator kwargs: forwarded into the long-range
  setup, not rejected. A misspelling can look accepted.
- Fine-tuning ignores use_ema, early_stopping_patience and
  stage_two_*. run_fine_tuning is a separate loop from
  run_training.

WILL NOT WORK
- Several GPUs for one simulation, EXCEPT flash with
  --flash-devices N (GPUs of one node). The neighbour-list engine is
  single-device; nothing runs across nodes.
- CLI MD in a periodic box shorter than 2 x lr_cutoff on any side
  (24 A at the default 12 A): refused. Lower --lr-cutoff and use
  --kspace-electrostatics pme plus kspace_dispersion: pme in a
  settings file (MD has no --kspace-dispersion flag), or use the
  ASE calculator.
- NPT on a non-periodic structure (it needs a cell). Fully
  anisotropic NPT coupling is rejected; semi-isotropic is
  experimental.
- Observable _grad / _jac suffixes outside So3lrCalculator.
- calculate_hessian together with calculate_stress, or with
  observable gradients.

K-SPACE HAS TWO SEPARATE ARGUMENTS
- kspace_electrostatics: calculator level. Builds the k-grid AND
  sets the solver. Requires kspace_smearing + kspace_spacing.
- coulomb_kspace_method: model level. Selects the solver only,
  default pme. An explicit value overrides the line above.
Use kspace_electrostatics unless you supply atoms.info['k_grid']
yourself.

LOG FILE
- Subcommands: named after --output with the extension REPLACED:
  traj.xyz -> traj.log, not traj.xyz.log.
- so3lr --settings run.yaml: named after the SETTINGS file,
  run.yaml -> run.log, unless log_file is set.
- Its header records the resolved model name, precision, cutoffs
  and thermostat settings, and an "Engine:" line names the MD
  engine.
```

## Where to check

- The Python `aux` dictionary uses the model's own key names, while
  `so3lr eval --targets` translates between `hirshfeld_ratios` and `a0_ratios`.
  See [Observables](observables.md#names-differ-between-model-generations).
- The log's `Engine:` line says which MD engine ran, so two runs being compared
  may not use the same one:

    ```text
    Engine: neighbour-list potential (flash cannot: it has no fixed atoms)
    ```

    With `--engine flash`, the run fails instead of falling back. See
    [Molecular dynamics](../usage/md.md).

- Warnings about unknown settings keys go to stderr before the log file is
  opened, so check the job's stderr, not the log:

    ```text
    Unknown setting 'md_t' in settings.yaml; ignoring. Did you mean: md_dt, md_T, md_P?
    ```

## Also worth giving your agent

- [Settings file reference](../usage/settings.md): every YAML key with its
  default
- [Command line](../usage/cli.md): every flag, exit codes and the SLURM section
- [Molecular dynamics](../usage/md.md): the two engines, restarts and the
  box-size rule
- [Observables](observables.md): which names each interface accepts
- [Long-range electrostatics](electrostatics.md#the-two-spellings): the two
  k-space arguments

If you are asking an agent to modify SO3LR itself, point it at `AGENTS.md` and
`DEVELOPERS.md` in the repository and the
[contributing guide](../contributing.md) instead.
