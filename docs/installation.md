# Installation

Try SO3LR without any local installation using our
[Colab notebook :material-open-in-new:](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb){ target=_blank }.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) 0.11.26 or newer

SO3LR can be used either with CPU or with GPU. It runs much faster on GPU.

## Install with uv

We recommend installing SO3LR with [uv](https://docs.astral.sh/uv/) (it's
magical) from a checkout, using the committed lockfile:

```shell
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
uv sync --locked --extra cuda12   # GPU; drop --extra cuda12 for CPU
source .venv/bin/activate
so3lr --help
```

The models are bundled with the package, so there is nothing else to download.
More details about JAX installation can be found [here](https://docs.jax.dev/en/latest/installation.html).

## Install with pip

If JAX is already installed on your cluster (CUDA or ROCm), install SO3LR into
that environment:

```shell
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
pip install .
```

Without an existing JAX, `pip install .` installs the CPU build. For an NVIDIA
GPU, use `pip install ".[cuda12]"`.

SO3LR excludes JAX 0.10.2, which computes wrong energies and forces on the CPU
without an error. An existing 0.10.2 install is moved to 0.10.1. On a GPU,
install with the `cuda12` extra so that JAX's CUDA plugin moves with it;
otherwise JAX finds a plugin it cannot use and runs on the CPU.

## CUDA kernels

On an NVIDIA GPU, the flash engine's fastest path runs on
[CUDA kernels](usage/md.md#cuda-kernels), which SO3LR builds with `nvcc` on
first use. `nvcc` comes with the CUDA toolkit, not with JAX. SO3LR looks for it
in `SO3LR_NVCC` (the full path to `nvcc`), then `$CUDA_HOME/bin`, the `PATH`
and `/usr/local/cuda/bin`. Without it, flash runs on its Pallas kernels: the
log has no `Flash: CUDA kernels (...)` line, and with a `--flash-mode` given
it says `Flash: CUDA kernels not used (nvcc not found (set SO3LR_NVCC or
CUDA_HOME)); Pallas kernels`.

## Verify it works

Run a short calculation to check that JAX works and which device it uses:

```shell
python - <<'EOF'
import jax
from ase import Atoms
from so3lr import So3lrCalculator

print("backend:", jax.default_backend(), jax.devices())

atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
atoms.calc = So3lrCalculator(model="so3lr-tauri-m", lr_cutoff=100)

print("energy:", atoms.get_potential_energy(), "eV")
print("max force:", abs(atoms.get_forces()).max(), "eV/A")
EOF
```

If `backend:` prints `cpu` on a GPU machine, see
[troubleshooting](#troubleshooting).

## Optional extras

| Extra | What it adds |
|---|---|
| `cuda12` | CUDA 12 builds of JAX, Linux x86-64 only. |
| `pysages` | [PySAGES](https://github.com/SSAGESLabs/PySAGES) for [enhanced sampling](advanced/enhanced-sampling.md). |
| `tfds` | TensorFlow Datasets loader for [training](training/from-scratch.md#tfds-datasets) and fine-tuning. |
| `wandb` | [Weights & Biases logging](training/from-scratch.md#weights-biases-logging) for training. |
| `matscipy` | A faster neighbour search for the ASE calculator in small or skewed periodic cells. |

=== "uv"

    ```shell
    uv sync --locked --extra cuda12 --extra pysages
    ```

=== "pip"

    ```shell
    pip install ".[cuda12,pysages]"
    ```

## Development install

```shell
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
uv sync --locked --group dev
uv run pytest tests/
```

See [Contributing](contributing.md) for the test layout and the lockfile.

## Troubleshooting

??? failure "Install fails with a resolver error mentioning `python`"
    Your interpreter is outside 3.12–3.13. Name a supported one, and uv fetches
    it if needed:

    ```shell
    uv sync --locked --python 3.12
    ```

??? failure "uv refuses the project with a `required-version` error"
    Your uv is older than 0.11.26. Upgrade it with `uv self update` and sync
    again.

??? failure "JAX only sees the CPU on a GPU node"
    The CUDA build of JAX is missing or does not match the driver. Check with:

    ```shell
    python -c "import jax; print(jax.default_backend(), jax.devices())"
    ```

    Then run `uv sync --locked --extra cuda12`, or `pip install ".[cuda12]"`.

??? failure "Out of memory on GPU"
    JAX reserves most of the GPU memory at start-up, which often causes this
    on a shared node. To allocate on demand instead:

    ```shell
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    ```

    For systems that are too large for the card, see
    [Molecular dynamics](usage/md.md).

??? failure "Importing `jax_md` fails inside `flax`, or PME hangs on CPU"
    JAX was upgraded past SO3LR's supported versions, or an older Flax
    (0.12.0 or earlier) was kept. Fix it with `uv sync --locked`, or

    ```shell
    pip install "jax<0.11,!=0.10.2" "jaxlib<0.11,!=0.10.2" "flax>=0.12.1,<0.12.9"
    ```

    On a GPU, use `"jax[cuda12]<0.11,!=0.10.2"` in place of the first, so that
    the CUDA plugin follows.
