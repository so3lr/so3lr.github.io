# Fine-tuning

SO3LR can be fine-tuned on custom datasets to improve performance on specific
systems or molecular environments. Fine-tuning is still experimental, so its
interface may change between releases.

## Data requirements

The input datafile should be readable by ASE (e.g., `.extxyz`, `.xyz`) or be a
`.npz` file, and include atomic positions and forces. We strongly recommend
also including dipole moments and polarizability/Hirshfeld ratios (see
[Choosing a strategy](#choosing-a-strategy)). Units are eV and Å by default.
Please refer to the [Zenodo repository](https://doi.org/10.5281/zenodo.14779793)
for an example data file.

## Basic run

```shell
so3lr finetune --datafile dataset.xyz --workdir so3lr_finetuned \
    --num-train 1000 --num-valid 100
```

| Flag | Meaning |
|---|---|
| `--datafile` | Training data. |
| `--workdir` | Output directory for checkpoints and logs. |
| `--num-train` | Number of training structures (or set `training.num_train`). |
| `--num-valid` | Number of validation structures (or set `training.num_valid`). |
| `--strategy` | Which parameters to train (default `full`). See below. |
| `--config` | YAML file that replaces the defaults in `so3lr/config/finetune.yaml`. |
| `--model` | Starting point, a bundled name or a previous fine-tune's workdir. Default `so3lr-tauri-m-2606`. |
| `--log-file` | Log path. Default `<workdir>/finetuning.log`. |

The split is random, seeded by `data.split_seed`, and saved to
`<workdir>/data_splits.json`.

To fine-tune on a [TFDS dataset](from-scratch.md#tfds-datasets), list it under
`data.datasets` in your `--config`. You still have to pass an existing
`--datafile` and `--num-train` / `--num-valid`, although they are not used.

To use the fine-tuned model, pass its directory as the model:

```shell
so3lr nvt --input geometry.xyz --model ./so3lr_finetuned
```

## Choosing a strategy

!!! warning "Fine-tuning can silently damage the long-range terms"
    On energies and forces alone, every strategy can degrade the long-range
    behaviour. Strategies that train the representation, `freeze_long_range`
    included, shift the charges, polarizabilities and C6 coefficients, and the
    others can still worsen interactions outside the training data. The forces
    still look fine. We are working on a fix.

To keep the long-range heads themselves fixed, use `freeze_long_range`:

```shell
so3lr finetune --datafile forces_only.xyz --strategy freeze_long_range \
    --num-train 1000 --num-valid 100
```

`freeze_long_range` (alias `forces_only`) freezes the partial-charge,
`a0_ratios` and `c6_ratios` heads, and trains the rest. If your dataset
includes dipole, Hirshfeld or C6 labels, we suggest `full`, or adding those
terms to `loss_weights`.

### Recommended

| Strategy | Train | When |
|---|---|---|
| `full` | Everything | Dataset has dipole/ratio labels. |
| `freeze_long_range` (`forces_only`) | Representation + energy readout | Forces-only datasets. |
| `last_layer_and_final_mlp` | Last message-passing layer + all of `observables_0`, partial charges included | Small datasets with dipole labels, or when `full` overfits. |
| `ll_fmlp` | Last message-passing layer + the energy readout only | Small forces-only datasets. |

### All strategies

`full` and `freeze_long_range` train everything except what they freeze. All
other strategies train only the listed parameters.

| Group | Holds |
|---|---|
| `layers_0` … `layers_{n-1}` | The message-passing layers |
| `observables_0` | The energy readout and the partial-charge network (`observables_0/electrostatic_energy`) |
| `observables_1` | The dipole, computed from the partial charges (no parameters) |
| `observables_2` | Polarizability (`a0_ratios`) / Hirshfeld ratios |
| `observables_3` | C6 ratios (`so3lr-tauri-s/m/l` only) |

| Strategy | Trains |
|---|---|
| `full` | Everything |
| `freeze_long_range` / `forces_only` | Everything except `observables_0/electrostatic_energy`, `observables_2`, `observables_3` |
| `obs_0` | All of `observables_0` |
| `final_mlp` / `freeze_all_but_final_mlp` | Only the energy readout inside `observables_0` |
| `hirshfeld` | `observables_2` only |
| `obs_3` | `observables_3` only |
| `obs_23` | `observables_2` and `observables_3` |
| `obs_023` | `observables_0`, `_2` and `_3` |
| `final_mlp_and_hirshfeld` | `observables_0` and `observables_2` |
| `last_layer` | Last message-passing layer only |
| `first_layer` | First message-passing layer only |
| `first_layer_and_last_layer` | Both outer message-passing layers |
| `last_layer_and_final_mlp` | Last message-passing layer + all of `observables_0` |
| `ll_fmlp` | Last message-passing layer + the energy readout inside `observables_0` |
| `ll_obs_023` | Last message-passing layer + `observables_0`, `_2` and `_3` |

Every strategy that trains the whole `observables_0` also trains the
partial-charge network. On forces-only data, use `final_mlp`, `ll_fmlp` or
`freeze_long_range`.

## Custom configuration

```shell
so3lr finetune --datafile dataset.xyz --workdir out --config custom_finetune.yaml
```

The default `so3lr/config/finetune.yaml` uses AdamW (learning rate 1e-4 with
exponential decay), `batch_max_num_graphs: 8`, validation every 10,000 steps
and forces-only loss weights. Copy it as a starting point: the file you pass
replaces the default instead of being merged into it.

The architecture is taken from the checkpoint, so a `model:` block is ignored
with a warning. Long-range pairs are built up to `data.neighbors_lr_cutoff`
(100 Å in `finetune.yaml`); lower it for large periodic cells.

With a TFDS dataset, the [data filters](from-scratch.md#data-filters) apply
and `training.batch_max_num_nodes` / `batch_max_num_edges` (plus
`batch_max_num_pairs` with long-range pairs) must be set, and `data.shift_mode`
must stay `null`.

Some [optional training features](from-scratch.md#optional-training-features)
are not available when fine-tuning. Setting an ignored key has no effect and
gives no warning.

| Feature | Fine-tuning |
|---|---|
| `reduce_lr_on_plateau` (+ `reduce_lr_factor`, `reduce_lr_patience`, `reduce_lr_min_scale`, `reduce_lr_min_delta`, `reduce_lr_cooldown`) | ✅ honoured |
| [Robust loss](from-scratch.md#robust-loss) (`use_robust_loss`, `use_adaptive_robust_loss`, …) | ✅ honoured (off in `finetune.yaml`) |
| [W&B logging](from-scratch.md#weights-biases-logging) (`use_wandb`) | ✅ honoured |
| `use_ema` / `ema_decay` | ❌ ignored |
| `early_stopping_patience` | ❌ ignored |
| `stage_two_start_epoch` / `stage_two_loss_weights` | ❌ ignored |

## Continuing a fine-tune

To start a new round of fine-tuning from a previous one, point `--model` at its
workdir:

```shell
so3lr finetune --datafile more_data.xyz --workdir round2 \
    --model ./so3lr_finetuned --strategy full
```

Re-running into the same `--workdir` instead resumes from its checkpoints (see
[Restarts](from-scratch.md#restarts)).

## Checking the result

We recommend always evaluating the fine-tuned model on held-out data, and
checking more than just the forces:

```shell
so3lr eval --datafile test.xyz --model ./so3lr_finetuned \
    --targets forces,energy,dipole_vec,a0_ratios --save-to test_pred.xyz
```

If the dipole or ratio metrics are worse than those of the base model, the
long-range heads were disturbed and you should revisit your strategy.
