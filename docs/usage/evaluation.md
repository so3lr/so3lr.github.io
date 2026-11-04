# Dataset evaluation

Evaluate a SO3LR model on a dataset with `so3lr eval`:

```shell
so3lr eval --datafile dataset.extxyz --batch-size 1 \
    --lr-cutoff 100 --save-to predictions.extxyz
```

The input can be any file that is digestible by
[`ase.io.iread`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.iread).
The command will collect and print metrics on the dataset and save the
predictions to the specified output file.

To evaluate a different bundled model, add `--model so3lr-tauri-s` (or another
[model name](../models.md)). Without it, the default `so3lr-tauri-m-2606` is
used.

Labels are assumed to be in eV and Å. The default long-range cutoff of 12 Å is
meant for periodic, condensed-phase systems. For gas-phase molecules, use
`--lr-cutoff 100`, otherwise electrostatics and dispersion are truncated.

The legacy `so3lr` model was not trained on energies, so compare only energy
differences there.

The charge and spin of each structure come from the `charge=` and
`multiplicity=` keys in its info, and default to a neutral singlet. If
`multiplicity=` is absent, `spin=` is read as the multiplicity.
`--total-charge` and `--spin` override these values for every structure.

## What gets predicted

| Property | Notes |
|---|---|
| `energy` | eV. Relative only for the legacy `so3lr` model. |
| `forces` | eV/Å. |
| `dipole_vec` | Molecular dipole vector. |
| ratio head | `a0_ratios` for `so3lr-tauri-s/m/l`, `hirshfeld_ratios` for the legacy `so3lr` model. |
| `c6_ratios` | Per-atom C6-coefficient ratios. `so3lr-tauri-s/m/l` only. |

Energy and forces are assumed to be present in the datafile. The other labels
are optional; if they are missing, their metrics are `NaN` and the run
continues. The metrics are printed as a table of MAE, RMSE and count per target.

`--targets` accepts either ratio-head name and maps it to the one the selected
model emits, so a `so3lr-tauri-m` run reports `a0_ratios_rmse`. See
[Names differ between model generations](../advanced/observables.md#names-differ-between-model-generations).

## Observable targets

Intermediate model quantities need no ground truth, so they can be added to
`--targets` as well. They are written to the output file alongside the predictions:

```shell
so3lr eval --datafile dataset.extxyz --lr-cutoff 100 --save-to predictions.extxyz \
    --targets forces,dipole_vec,nn_energy,partial_charges,electrostatic_energy,dispersion_energy
```

Every name marked *eval* in the
[observables table](../advanced/observables.md#available-observables) works
here.

## Atom embeddings

The final-layer per-atom features (`x`) are saved to a separate compressed
NumPy file, `<output>_x.npz` (or `<datafile>_x.npz` without `--save-to`). To
save them, add `x` to `--targets`:

```shell
so3lr eval --datafile dataset.extxyz --lr-cutoff 100 --save-to predictions.extxyz \
    --targets forces,x
# writes: predictions.extxyz   (standard predictions)
#         predictions_x.npz    (atom embeddings)
```

```python
import numpy as np

data = np.load("predictions_x.npz")
embeddings = data["embeddings"]   # (total_atoms, features)
n_atoms = data["n_atoms"]         # (n_structures,)

# split back into per-structure arrays
per_structure = np.split(embeddings, np.cumsum(n_atoms[:-1]))
```

## Output files

Existing outputs are replaced only with `--overwrite`. `--save-to` must end in
`.xyz` or `.extxyz`.

## Analyzing predictions

The predictions can be analyzed in Python. Each predicted property is stored
with a `_so3lr` suffix:

```python
import numpy as np
from ase.io import iread

true, pred = [], []
for a in iread("predictions.extxyz"):
    true.append(a.get_forces())
    pred.append(a.arrays["forces_so3lr"])

rmse = np.sqrt(np.mean(np.square(np.stack(true) - np.stack(pred))))
print(rmse)
```

Per-structure scalars such as the energy are stored in `a.info` (e.g.
`a.info["energy_so3lr"]`), and per-atom arrays in `a.arrays`.

For more in-depth evaluation using Python, check out the
[evaluation notebook](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/evaluate_so3lr_on_dataset.ipynb).

## Periodic datasets

Periodic structures are evaluated in real space up to `--lr-cutoff`. Add
`--kspace-electrostatics` (and optionally `--kspace-dispersion`) for the
reciprocal-space term:

```shell
so3lr eval --datafile crystals.extxyz --save-to predictions.extxyz \
    --kspace-electrostatics pme --kspace-smearing 4.0 --kspace-spacing 0.5
```

In that case, every structure must be periodic in all three directions. See
[Long-range electrostatics](../advanced/electrostatics.md) for the parameters.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--datafile` | — | Dataset to evaluate. |
| `--save-to` | `None` | Output `.extxyz`. Metrics still print without it. |
| `--targets` | `forces,dipole_vec,hirshfeld_ratios` | Comma-separated properties. |
| `--batch-size` | `1` | Structures per batch. Raise it (16–64 on a GPU) for many small, similarly-sized structures. |
| `--model` | `so3lr-tauri-m-2606` | Bundled name or path. |
| `--lr-cutoff` | `12.0` | Use `100` for gas phase. |
| `--dispersion-damping` | `2.0` | Dispersion switches off from `lr_cutoff - dispersion_damping` Å. |
| `--precision` | `float32` | `float64` for tight comparisons. |
| `--total-charge`, `--spin` | from file | Override per-structure values. |
| `--charge-embed-total-charge` | follows the charge | Charge seen by the charge embedding only. |
| `--charge-equilibration-total-charge` | follows the charge | Charge that charge equilibration conserves to. |
| `--kspace-electrostatics` | off | `ewald` / `pme` / `p3m`, with `--kspace-smearing` (`4.0`), `--kspace-spacing` (`2.0`) and `--kspace-interp-nodes` (`4`). |
| `--kspace-dispersion` | off | `ewald` / `pme`, with `--kspace-dispersion-smearing` (`2.0`), `--kspace-dispersion-spacing` (`3.0`) and `--kspace-dispersion-interp-nodes` (`4`). |
| `--jit-compile` / `--no-jit-compile` | on | Disable when debugging shape errors. |
| `--overwrite` / `--no-overwrite` | refuse | Replace existing output files. |
| `--log-file` | `<save-to>` with `.log` | Console only when there is no `--save-to`. |

Batches are padded to the largest structure in the batch, so a dataset with
widely varying sizes may run faster at `--batch-size 1`.

## Related

- [Binding energies](../advanced/binding-energies.md): dimer interaction energies
- [Observables](../advanced/observables.md): what each intermediate quantity means
- [Fine-tuning](../training/finetuning.md): if the metrics are not good enough
