# Binding energies

You can calculate binding energies with SO3LR as reported in the
[JACS paper](https://doi.org/10.1021/jacs.5c09558) (Fig. 2B).

## The method

The binding energy is computed as the difference between the energies of the
bound dimer and the same dimer with the two monomers translated far apart
(beyond the long-range cutoff), with charges assigned to each monomer
separately to prevent spurious charge transfer:

```
E_bind = E_dimer - E_separated
```

Only the charge equilibration is done per monomer; the charge embedding sees
the dimer's total charge in both frames.

## Preparing structures

For XYZ files with dimer metadata (`charge_a`, `charge_b`, `selection_a`,
`selection_b`, like the
[NCIAtlas](https://github.com/Honza-R/NCIAtlas/blob/main/geometries/NCIA_D1200/1.01.01_100.xyz)
format), generate the original and translated structures:

```shell
python -m so3lr.prepare_dimer_xyz --datafile sapt10k.xyz
```

This writes `sapt10k_prepared.xyz` with alternating frames: dimer, separated,
dimer, separated, … Structures with elements the chosen model does not support
are skipped and written to `<input>_unsupported.xyz`.

| Option | Default | Meaning |
|---|---|---|
| `--datafile` | required | Input dimers, any file `ase.io` reads |
| `--output` | `<input>_prepared.xyz` | Output file (extxyz) |
| `--model` | `so3lr-tauri-m-2606` | Model whose element coverage is checked (`so3lr` for the legacy model) |

The selections are 1-based inclusive ranges (`selection_a=1-12`). Monomer A
must be the first atoms and monomer B the rest, and the two selections together
must cover every atom. If they do not, the per-monomer charges are silently not
applied.

## Evaluating

```shell
so3lr eval --datafile sapt10k_prepared.xyz --targets energy \
    --save-to sapt10k_eval.xyz --lr-cutoff 100
```

Use `--lr-cutoff 100` for these gas-phase dimers, and keep the default
`--batch-size 1`.

## Computing the binding energies

`so3lr eval` recognises these alternating dimer/separated files and prints the
binding-energy errors against `benchmark_Eint` (in kcal/mol), computed from the
full-precision energies:

```text
Binding energies (dimer - separated) for 20 dimers, against benchmark_Eint in kcal/mol:
  MAE 1.117 kcal/mol · RMSE 1.576 kcal/mol · max |error| 4.658 kcal/mol
```

`so3lr start` → **Evaluate a dataset** includes a 20-dimer SAPT10k example. To
work with the individual values:

```python
from ase.io import read

mols = read("sapt10k_eval.xyz", index=":")
energy = [a.info["energy_so3lr"] for a in mols]

# frames alternate dimer / separated
bind_energy = [energy[i] - energy[i + 1] for i in range(0, len(energy), 2)]
```

The binding energies are in eV, and negative values indicate binding.

!!! warning "Precision of the saved energies"
    `so3lr eval` writes `energy_so3lr` rounded to 6 significant figures, which
    for a large dimer can reach 0.01 eV (0.23 kcal/mol) per frame. When that
    matters, evaluate the frames with the [ASE calculator](../usage/ase.md)
    and take the difference at full precision.

The energy metrics that `so3lr eval` prints for these files are not
binding-energy errors: both frames carry the reference interaction energy
(`benchmark_Eint`) as their `energy` label. Compare `bind_energy` against
`benchmark_Eint` yourself, and check which units it is in.

## Data

The [SAPT10k](https://doi.org/10.1063/5.0204064) dataset is bundled under
`examples/data/sapt10k.zip`.
