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

On data with energies and forces only, use one of three recipes. Such labels
cannot tell short-range from long-range energy, so `--strategy full` on its own
lets the predicted charges, polarizabilities and C6 coefficients drift. The
recipes keep them. They are also described at the top of
`so3lr/config/finetune.yaml`.

| Recipe | Strategy | Long-range terms |
|---|---|---|
| Anchors (most accurate) | `full` + `training.long_range_anchor_weights` | Kept close to the starting model's on every batch |
| Exact long range | `readout_only` + a residual energy readout | Identical to the starting model's |
| Pseudo-labels | `full` + the starting model's predictions as labels | Kept close, as with anchors |

**Anchors.** A frozen copy of the starting model predicts the charges and
ratios of every training and validation batch, and the fine-tuned model is
trained to match them. In your copy of `finetune.yaml` (the file `--config`
passes replaces it as a whole):

```yaml
training:
  loss_weights:
    forces: 1.0
    energy: 1.0
  long_range_anchor_weights:
    partial_charges: 1.0
    a0_ratios: 1.0
    c6_ratios: 1.0
```

**Exact long range.** `readout_only` freezes the message-passing layers and the
long-range heads, and trains the energy readout plus a new residual readout
that you add to your copy of `finetune.yaml`:

```yaml
model:
  energy_residual_mlp_widths: [512, 512]
```

```shell
so3lr finetune --datafile dataset.xyz --workdir so3lr_readout \
    --config custom_finetune.yaml --strategy readout_only \
    --num-train 1000 --num-valid 100
```

The charges, ratios and dipoles stay exactly those of the starting model.
Without a residual readout, `readout_only` trains the energy readout alone, as
`final_mlp` does, and logs a warning that `observables_0/energy_residual*`
matches no parameter.

**Pseudo-labels.** This is the anchor recipe run offline, for example with a
different teacher model. Write the teacher's predictions into the data:

```shell
so3lr eval --datafile data.extxyz --save-to teacher.extxyz \
    --targets energy,forces,partial_charges,a0_ratios,c6_ratios
```

Copy the `partial_charges_so3lr`, `a0_ratios_so3lr` and `c6_ratios_so3lr`
arrays of `teacher.extxyz` into your data file as `partial_charges`,
`a0_ratios` and `c6_ratios`, and fine-tune with `--strategy full` and the
`partial_charges`, `hirshfeld_ratios` and `c6_ratios` loss weights set to 1.0.

With any recipe, the labels must include dispersion as SO3LR does. SO3LR keeps
its own dispersion term, so fitting energies without a dispersion correction
(plain DFT without D3, D4 or MBD) teaches every recipe to underbind
non-covalent complexes.

### Recommended

| Strategy | Train | When |
|---|---|---|
| `full` | Everything | Dataset has dipole/ratio labels, or with anchors or pseudo-labels. |
| `readout_only` | The energy readout + the residual readout | Energy/force-only data, when the long-range terms must not change. |
| `last_layer_and_final_mlp` | Last message-passing layer + all of `observables_0`, partial charges included | Small datasets with dipole labels, or when `full` overfits. |
| `ll_fmlp` | Last message-passing layer + the energy readout only | Small datasets, when `full` overfits. |

### All strategies

`full` trains everything. All other strategies train only the listed
parameters.

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
| `readout_only` | The energy readout and the residual readout (`observables_0/energy_residual*`) inside `observables_0` |

Every strategy that trains the whole `observables_0` also trains the
partial-charge network. Every strategy that trains a message-passing layer
changes the features the long-range heads read.
Only `final_mlp` and `readout_only` leave the long-range terms exactly as they
were. On energy/force-only data, use one of the
[three recipes](#choosing-a-strategy).

## Custom configuration

```shell
so3lr finetune --datafile dataset.xyz --workdir out --config custom_finetune.yaml
```

The default `so3lr/config/finetune.yaml` uses AdamW (learning rate 1e-4 with
exponential decay), `batch_max_num_graphs: 8`, validation every 10,000 steps
and forces-only loss weights. Copy it as a starting point: the file you pass
replaces the default instead of being merged into it.

If your data has energies, add `energy: 1.0` to `loss_weights`. Pretraining
used 0.01, but at 0.01 the energies barely train in a fine-tune; at 1.0 they
train without costing force accuracy in our tests.

The architecture is taken from the checkpoint. A `model:` block may set only
the residual readout and the [long-range settings for periodic
data](#periodic-data); other keys are ignored with a warning.

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
| [`data_parallel`](from-scratch.md#multiple-gpus) | ✅ honoured with a TFDS dataset |
| [Robust loss](from-scratch.md#robust-loss) (`use_robust_loss`, `use_adaptive_robust_loss`, …) | ✅ honoured (off in `finetune.yaml`) |
| [W&B logging](from-scratch.md#weights-biases-logging) (`use_wandb`) | ✅ honoured |
| `use_ema` / `ema_decay` | ❌ ignored |
| `early_stopping_patience` | ❌ ignored |
| `stage_two_start_epoch` / `stage_two_loss_weights` | ❌ ignored |

### Keys for energy- and force-only data

| Key | Meaning |
|---|---|
| `training.long_range_anchor_weights` | Loss weights that anchor predictions to a frozen copy of the starting model. Keys: `partial_charges`, `a0_ratios` (`hirshfeld_ratios` for the legacy `so3lr`), `c6_ratios` (not predicted by `so3lr`) and `dipole_vec` (optional). Each batch costs one extra forward pass, which made runs 2–6 % longer. |
| `model.energy_residual_mlp_widths` | Hidden widths of a residual per-atom energy readout added to the pretrained one, e.g. `[512, 512]`. Its last layer starts at zero, so the model's energy is unchanged at the first step. |
| `model.energy_residual_inputs` | What the residual readout reads: `last` (default) the final-layer features, `all_layers` the features after the embedding and after every message-passing layer. |
| `training.loss_weights.partial_charges` | Per-atom charges, read from a `partial_charges` array in the data file (see [Datasets](../reference/datasets.md#optional-labels)). |

The residual readout becomes part of the fine-tuned model. The flash engine
does not take a model with one, so MD and `eval` with it run on the
neighbour-list engine. It needs ASE or NPZ data; with a TFDS dataset it is refused. A strategy that freezes it, such as
`final_mlp` or `ll_fmlp`, logs a warning, because it would stay zero.
When you fine-tune a model that already has a residual readout, leave these
keys out or set the same values.

### Periodic data

Long-range pairs are built up to `data.neighbors_lr_cutoff` (100 Å in
`finetune.yaml`). For a molecule smaller than that, this list holds every
pair, but for a periodic
cell it stops at that distance, where the long-range terms would be cut off
abruptly instead of switched off smoothly as in MD and `eval`. Periodic ASE or
NPZ data is therefore refused unless `model.cutoff_lr` is set and is at most
`data.neighbors_lr_cutoff`. TFDS data is not checked: set both yourself. Set both to the same value and add the Coulomb tail
in k-space:

```yaml
data:
  neighbors_lr_cutoff: 12.0   # Å
model:
  cutoff_lr: 12.0             # Å
  kspace_electrostatics: pme  # or ewald / p3m
```

A fine-tune config may set `cutoff_lr`, `kspace_electrostatics`,
`kspace_smearing`, `kspace_spacing`, `kspace_dispersion`,
`kspace_dispersion_smearing` and `kspace_dispersion_spacing`. They apply to
training only and are recorded under `long_range` in
`<workdir>/fine_tuning.json`. The fine-tuned model keeps the checkpoint's own
values, so at inference you choose the cutoff and k-space as for any model.

With k-space on, the keys you leave unset take the library defaults: 4 Å
smearing, 2 Å spacing and no k-space dispersion (with `kspace_dispersion: true`,
2 Å smearing and 3 Å spacing for dispersion). These are the Python
calculators' defaults. To train with the settings `so3lr md` and `so3lr eval`
pick for a periodic structure by default, set `cutoff_lr` and
`neighbors_lr_cutoff` to 10 Å, `kspace_smearing: 2.0`, `kspace_spacing: 1.55`
and `kspace_dispersion: true`. Under k-space, every structure must be
periodic in all three directions; see [K-space during
training](from-scratch.md#k-space-during-training).

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
