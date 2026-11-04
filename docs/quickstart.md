# Quick start

Try SO3LR without any local installation using our
[Colab notebook :material-open-in-new:](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb){ target=_blank }.

## Install

SO3LR supports Python 3.12 and 3.13.

=== "uv"

    We recommend installing SO3LR with [uv](https://docs.astral.sh/uv/)
    (it's magical):

    ```shell
    git clone https://github.com/general-molecular-simulations/so3lr.git
    cd so3lr
    uv sync --locked --extra cuda12   # GPU; drop --extra cuda12 for CPU
    source .venv/bin/activate
    ```

=== "pip"

    If JAX is already installed on your cluster (CUDA or ROCm), install SO3LR
    into that environment:

    ```shell
    git clone https://github.com/general-molecular-simulations/so3lr.git
    cd so3lr
    pip install .
    ```

See [Installation](installation.md) for optional extras.

## Your first simulation

Run a 10 ps NVT (constant volume and temperature) simulation at 300 K on any
geometry that ASE can read:

```shell
so3lr nvt --input geometry.xyz --temperature 300 \
    --dt 0.5 --md-cycles 100 --md-steps 200
```

The run has `md-cycles` × `md-steps` steps, so 100 × 200 × 0.5 fs = 10 ps. It
writes:

- `geometry_nvt.xyz`: the trajectory;
- `geometry_nvt.log`: energies, temperature and timings;
- `geometry_nvt_opt.xyz`: the relaxation run before MD.

A large enough periodic box on a recent NVIDIA GPU runs on the fast **flash**
engine, and anything else on the neighbour-list engine (see
[Molecular dynamics](usage/md.md)).

The default long-range cutoff of 12 Å is meant for periodic, condensed-phase
systems; for gas-phase simulations, we suggest using `--lr-cutoff 100`.

Geometry optimization and the other ensembles work the same way:

```shell
so3lr opt --input geometry.xyz --force-conv 0.05      # geometry optimization
so3lr npt --input box.xyz --temperature 300 --pressure 1.0
so3lr nve --input geometry.xyz --temperature 300
```

Each command can be run with `--help` to see all available options:

```shell
so3lr --help          # command overview
so3lr --help-full     # every command with examples
so3lr nvt --help      # flags for one command
```

## Energies and forces in Python

To use SO3LR as an ASE calculator:

```python
from ase.io import read
from so3lr import So3lrCalculator

atoms = read("geometry.xyz")
atoms.info["charge"] = 0          # total charge; 0 if unset

atoms.calc = So3lrCalculator(model="so3lr-tauri-m", lr_cutoff=100)

print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())            # eV/Å
```

To use `dtype=np.float64`, enable JAX's 64-bit mode before importing SO3LR
(see the [API reference](reference/api.md)).

## Longer runs

For production runs, we recommend a settings file. It also gives access to
features that have no CLI flag, such as frozen atoms and enhanced sampling:

```yaml title="settings.yaml"
ensemble: nvt
input_file: geometry.xyz
output_file: traj.hdf5
model: so3lr-tauri-m

md_T: 300
dt: 0.5
md_cycles: 1000
md_steps: 1000

restart_save_path: checkpoint.npz
```

Then run it with:

```shell
so3lr --settings settings.yaml
```

See [Molecular dynamics](usage/md.md) for restarts, constraints and output, and
the [Settings reference](usage/settings.md) for all available keys.

## Next steps

- [Models](models.md): which model to use, and how to pin one
- [Command line](usage/cli.md): every command and flag
- [Examples](examples.md): runnable notebooks
- [FAQ](reference/faq.md): cutoffs, charges, precision, common errors
