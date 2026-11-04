# Datasets

The quantum-mechanical datasets used to train and test SO3LR Tauri are on
Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22920714.svg)](https://doi.org/10.5281/zenodo.22920714)

<https://doi.org/10.5281/zenodo.22920714>

The data behind the legacy `so3lr` model and the
[JACS paper](https://doi.org/10.1021/jacs.5c09558) are at
<https://doi.org/10.5281/zenodo.14779793>.

Both are a starting point for [fine-tuning](../training/finetuning.md) or
[training from scratch](../training/from-scratch.md).

<!-- TODO (needs input): describe what the archive actually contains, so people
     can tell before a multi-GB download whether their chemistry is covered.
     Suggested columns:

     | Subset | Systems | Structures | Level of theory | Labels |
     |---|---|---|---|---|

     Plus: total download size, file format (extxyz / TFDS), whether the
     train/validation/test split is included or must be regenerated, and
     whether the 2606 models were trained on the same data as v1. -->

The bundled `so3lr-tauri-s/m/l` checkpoints record which data trained them. In
each model directory under `so3lr/models/`:

- `hyperparameters.yaml` (and `.json`): the training mixture under
  `data.datasets`, with each source's path on the training cluster and its
  sampling weight.
- `fine_tuning.json`: the run the checkpoint was refined from.
- `data_splits.json`: which loader built each split.

## Conventions

| | |
|---|---|
| Energy | eV |
| Length | Ångström |
| Forces | eV/Å |

The original `so3lr` model was not trained on energies, so with it only energy
differences between structures of the same composition are meaningful.

## Optional labels

Besides energies and forces, datasets can include these optional labels:

- `dipole_vec`: molecular dipole vectors
- `hirshfeld_ratios` / `a0_ratios` / `hirsh_ratios`: volume or
  polarizability ratios, mapped to the key the selected model uses
- `c6_ratios`: per-atom C6 coefficient ratios

Missing labels give `NaN` metrics in [evaluation](../usage/evaluation.md) and
are skipped in training.

Structures with elements a model does not cover are rejected with an error (see
[Models](../models.md#the-current-family)).

## Bundled sample data

Small inputs for the [example notebooks](../examples.md#bundled-data) ship with
the repository under `examples/data/`.
