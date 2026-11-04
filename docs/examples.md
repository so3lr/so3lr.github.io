# Examples

For simulations, we recommend the [command line interface](usage/cli.md),
which handles neighbour-list reallocation, checkpointing and logging, and can
use the faster [flash engine](usage/md.md). The notebooks here are for when you
want to understand how SO3LR works underneath, or build it into your own code.

The notebooks live in the
[`examples/` directory](https://github.com/general-molecular-simulations/so3lr/tree/main/examples)
of the repository. To run them locally, install the notebook dependencies with
`uv sync --locked --group notebooks`.

We suggest starting with the
[Colab notebook :material-open-in-new:](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb){ target=_blank },
which needs no local installation and runs on a free GPU.

## All notebooks

| Notebook | What it covers |
|---|---|
| [`so3lr_colab_example.ipynb`](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb) | Installs SO3LR in Colab and runs the CLI (`opt` / `nvt` / `npt` / `nve` / `eval`) from interactive forms, with a trajectory viewer. |
| [`nvt_jaxmd.ipynb`](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/nvt_jaxmd.ipynb) | Nosé–Hoover NVT through JAX-MD for a molecule in vacuum (DHA). |
| [`npt_jaxmd.ipynb`](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/npt_jaxmd.ipynb) | The same in a periodic box with a fluctuating cell. |
| [`production_run.ipynb`](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/production_run.ipynb) | Nosé–Hoover NPT arranged as a production run, with buffered HDF5 output in the layout the CLI writes. |
| [`evaluate_so3lr_on_dataset.ipynb`](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/evaluate_so3lr_on_dataset.ipynb) | Batched evaluation over a dataset with `jraph.dynamically_batch`, which avoids JIT recompilation. |

## Bundled data

`examples/data/` contains the small input files the notebooks use:

| File | Contents |
|---|---|
| `ala15_folded.xyz` | A folded 15-residue alanine peptide, a good biomolecular test case. |
| `ethanol.npz` | Ethanol trajectory data. |
| `sapt10k.zip` | The [SAPT10k](https://doi.org/10.1063/5.0204064) dimer dataset, used for [binding energies](advanced/binding-energies.md). |

Larger training and test datasets are available on
[Zenodo](https://doi.org/10.5281/zenodo.22920714) (see [Datasets](reference/datasets.md)).
