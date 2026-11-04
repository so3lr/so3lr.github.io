# Training from scratch

You can train a SO3krates/SO3LR model from scratch using a YAML (or JSON)
configuration file:

```shell
so3lr train --config config.yaml
```

The default configuration in `so3lr/config/config.yaml` includes settings for
the model architecture, optimizer, learning-rate schedule, batch size, loss
weights and data filtering. We suggest starting from it and overriding what you
need. The log is written to `<config>.log` unless you set `--log-file`.

!!! warning "The default config trains a plain SO3krates"
    `config.yaml` has the long-range terms switched off. For a SO3LR-style
    model with electrostatics and dispersion, set:

    ```yaml
    data:
      neighbors_lr_bool: true        # build long-range neighbour lists
      neighbors_lr_cutoff: 12.0      # Å; required once neighbors_lr_bool is on
    model:
      electrostatic_energy_bool: true
      dispersion_energy_bool: true
      zbl_repulsion_bool: true       # already the default
    ```

The default optimizer is Muon. Its `optimizer_args` are Muon-specific, so
clear or replace them when you change `optimizer.name`.

Training from scratch needs a large dataset and a lot of compute. If you want
SO3LR to work better on your own systems, [fine-tuning](finetuning.md) is
usually the better choice.

## Data and splits

`data.filepath` takes any ASE-readable file, a `.npz` file, or a folder of
either. TFDS datasets go under [`data.datasets`](#tfds-datasets) instead.
Units are set by `data.energy_unit`, `data.length_unit` and
`data.dipole_vec_unit` (default eV and Å).

`training.num_train` + `training.num_valid` structures are drawn at random,
seeded by `data.split_seed`. The rest form the test set, and all three index
lists are written to `<workdir>/data_splits.json`. Per-element energy shifts
can be subtracted with `data.shift_mode` (`null`, `mean`, `lse`, or `custom`
with `data.energy_shifts`).

### TFDS datasets

Install the extra with `uv sync --locked --extra tfds` (or
`pip install ".[tfds]"`). A TFDS dataset is a prepared `tensorflow_datasets`
directory with its own `train` and `validation` splits. SO3LR does not ship
the converter that builds one.

The TFDS loader is used for datasets listed under
[`data.datasets`](#multiple-datasets), even if there is only one. Compared with
an ASE source:

- The per-dataset `num_train` / `num_valid` take the first *N* records of each
  split. There is no random split, and `training.num_train` / `num_valid` are
  not used.
- `training.batch_max_num_nodes` and `batch_max_num_edges` must be set, plus
  `batch_max_num_pairs` when long-range pairs are on.
- Energy shifts are not supported, so keep `data.shift_mode: null`.
- `training.batch_n_proc` sets the number of parallel batch workers.

## Multiple datasets

Instead of a single `data.filepath`, you can list several datasets under
`data.datasets`. The trainer then samples from them according to the
per-dataset `weight` values:

```yaml
data:
  filepath: null
  datasets:
    - path: /data/dataset_a
      weight: 1.0
      # num_train: 5000     # optional per-dataset cap on the train split
      # num_valid: 500      # optional per-dataset cap on the validation split
    - path: /data/dataset_b
      weight: 0.5
```

Every path under `data.datasets` must be a [TFDS dataset](#tfds-datasets).
Weights are draw probabilities per dataset, not per record, so with equal
weights a small dataset is sampled more often per record than a large one.
Setting a dataset's weight to its selected size visits it about once per
epoch. Validation visits every record once and ignores the weights.

### Data filters

```yaml
data:
  filter:
    min_distance: 0.5   # Å
    max_force: 25.      # eV/Å
```

`max_force` drops structures with a force component above the value (or
non-finite forces), and `min_distance` drops structures with two atoms closer
than the value. Both are applied only by the TFDS loader; with an ASE or NPZ
source they are ignored with a warning in the log. `finetune.yaml` uses
`min_distance: 0.25`.

## Robust loss

The robust loss is on by default: `config.yaml` uses an adaptive
[general robust loss](https://arxiv.org/abs/1701.03077) instead of plain L2.
Its shape parameter α gives L2 at α = 2, Charbonnier at α = 1 and Cauchy at
α = 0. Lower α is more tolerant of outliers.

=== "Adaptive (default)"

    α and the scale are learned per target during training.

    ```yaml
    training:
      use_adaptive_robust_loss: true
      adaptive_robust_loss_init_alpha: 1.0   # initial α, range [0, 2]; 1.5 if the key is absent
      adaptive_robust_loss_init_scale: 1.0   # initial scale, must be > 0
    ```

=== "Fixed α"

    α is held constant at the value you set.

    ```yaml
    training:
      use_adaptive_robust_loss: false
      use_robust_loss: true
      robust_loss_alpha: 1.0
    ```

=== "Plain L2"

    ```yaml
    training:
      use_adaptive_robust_loss: false
      use_robust_loss: false
    ```

`use_robust_loss` and `use_adaptive_robust_loss` are mutually exclusive.
Validation can use its own settings through `use_robust_loss_validation` and
`robust_loss_alpha_validation`, which default to the training values. Robust
loss is also honoured when [fine-tuning](finetuning.md#custom-configuration).

## Optional training features

These features are all off by default (see the commented entries in
`so3lr/config/config.yaml`). Evaluations happen every
`training.eval_every_num_steps` gradient steps, and the patience counts below
are in evaluations.

### EMA of weights

```yaml
training:
  use_ema: true
  ema_decay: 0.999
```

This keeps an exponential moving average of the parameters, which is used for
validation and saved in the checkpoint.

### Early stopping

```yaml
training:
  early_stopping_patience: 20
  early_stopping_min_delta: 1.0e-4   # default
```

Training stops once the validation loss has not improved by more than
`early_stopping_min_delta` for that many consecutive evaluations.

### Reduce LR on plateau

```yaml
training:
  reduce_lr_on_plateau: true
  reduce_lr_factor: 0.75      # default
  reduce_lr_patience: 10      # default 1000
  reduce_lr_min_scale: 0.01   # default 0.0
  reduce_lr_min_delta: 1.0e-4 # default
  reduce_lr_cooldown: 0       # default
```

The learning rate is multiplied by `reduce_lr_factor` whenever the validation
loss has not improved by more than `reduce_lr_min_delta` for
`reduce_lr_patience` evaluations, down to `learning_rate × reduce_lr_min_scale`,
with `reduce_lr_cooldown` evaluations of waiting after each reduction. It works
on top of `learning_rate_schedule`, and is also honoured when
[fine-tuning](finetuning.md#custom-configuration).

### Two-stage loss reweighting

```yaml
training:
  stage_two_start_epoch: 200
  stage_two_loss_weights:
    energy: 0.1
    forces: 10.0
```

This switches to a second set of loss weights partway through training. Both
keys are needed.

## K-space during training

For periodic training data, the model can be trained with reciprocal-space
long range:

```yaml
model:
  kspace_electrostatics: pme   # pme / p3m / ewald; null (default) disables
  kspace_smearing: 2.0         # Å
  kspace_spacing: 1.0          # Å
```

The method applies to electrostatics and dispersion alike. A single grid is
sized from the first periodic cell, so keep the data fully periodic and the
cells similar in size; a non-periodic structure in the mix is not caught. See
[Long-range electrostatics](../advanced/electrostatics.md) for the parameters.

## Restarts

With `training.allow_restart: true` (the default), training resumes from the
checkpoints in an existing workdir. With `false`, an existing checkpoint
directory is refused. What a resume restores depends on the loader:

- **TFDS**: the full training state (optimizer, EMA, epoch, counters, RNG
  key) is restored, and the run continues with the next epoch. Dataset
  settings are not checked, so keep the data directories unchanged while a run
  is resumable.
- **ASE / NPZ**: the run continues from the weights and step only, with a
  fresh optimizer, EMA and data order.

## Weights & Biases logging

W&B logging is off by default. To use it, install the extra with
`uv sync --locked --extra wandb` (or `pip install ".[wandb]"`) and enable it in
your config:

```yaml
training:
  use_wandb: true
  wandb_init_args:          # passed to wandb.init()
    project: my-project
```

This sends training metrics and the full configuration, file paths included,
to Weights & Biases. Keep your credentials in the W&B login or environment,
never in the YAML file. `log_gradient_values: true` also logs gradient norms.

## Monitoring

Validation metrics are written to the log at every evaluation. Checkpoints are
saved under `checkpoints/` in the workdir, and the three with the lowest
validation loss are kept. You can pass the workdir wherever a model is
accepted; loading restores the most recent kept checkpoint:

```shell
so3lr eval --datafile test.xyz --model ./my_training_workdir
```

## Roadmap

Coming soon:

- Multi-head fine-tuning

If you would like to see new features, add your own functionality, or share
ideas, please reach out or
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues).
We would appreciate if you report any errors or inconsistencies.
