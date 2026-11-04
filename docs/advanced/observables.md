# Observables

SO3LR's total energy is a sum of physically distinct contributions, and the
model computes several per-atom quantities on the way. You can extract all of
them for analysis and method development.

## From Python

To get them from the ASE calculator, set `has_aux=True` and list the
observables you want:

```python
from jax import config

# Needed for dtype=np.float64; must run before SO3LR is imported.
config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms
from so3lr import So3lrCalculator

atoms = Atoms("OH2", positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
atoms.info["charge"] = 0.0

calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=100,
    dtype=np.float64,
    has_aux=True,
    observables=[
        "nn_energy",             # neural-network energy
        "electrostatic_energy",  # electrostatics
        "dispersion_energy",     # dispersion
        "nlh_repulsion",         # short-range repulsion
        "partial_charges",       # per-atom charges
        "a0_ratios",             # polarizability-volume ratios
    ],
)

atoms.calc = calc
energy = atoms.get_potential_energy()

aux = calc.results["aux"]

print(f"Total:         {energy:.6f} eV")
print(f"  NN:          {np.sum(aux['nn_energy']):.6f} eV")
print(f"  Electrostat: {np.sum(aux['electrostatic_energy']):.6f} eV")
print(f"  Dispersion:  {np.sum(aux['dispersion_energy']):.6f} eV")
print(f"  Repulsion:   {np.sum(aux['nlh_repulsion']):.6f} eV")
print(f"Partial charges: {aux['partial_charges']}")
print(f"Polarizability volume ratios: {aux['a0_ratios']}")
```

Energy-like observables, including `energy` in `aux`, are per-atom arrays of
shape `(n_atoms,)`. Sum over atoms to get the total contribution.

## Available observables

The three interfaces accept slightly different sets of names:

- **aux**: `calc.results["aux"]` with `has_aux=True`
- **eval**: `so3lr eval --targets ...`
- **MD**: `so3lr nvt --observable ...` / `observables:` in a settings file

| Observable | Description | Shape | Available in |
|---|---|---|---|
| `nn_energy` | Neural-network energy contribution | `(n_atoms,)` | aux · eval · MD |
| `electrostatic_energy` | Damped (real-space) electrostatic energy | `(n_atoms,)` | aux · eval · MD |
| `dispersion_energy` | Damped dispersion energy | `(n_atoms,)` | aux · eval · MD |
| `electrostatic_energy_undamped` | Undamped electrostatic energy | `(n_atoms,)` | aux · eval · MD |
| `dispersion_energy_undamped` | Undamped dispersion energy | `(n_atoms,)` | aux · eval · MD |
| `nlh_repulsion` | NLH short-range repulsion (`so3lr-tauri-s/m/l` only) | `(n_atoms,)` | aux · eval · MD |
| `zbl_repulsion` | Short-range repulsion: ZBL on legacy `so3lr`, NLH on `so3lr-tauri-s/m/l` | `(n_atoms,)` | aux · eval · MD |
| `partial_charges` | Atomic partial charges | `(n_atoms,)` | aux · eval · MD |
| `x` | Final-layer per-atom embeddings | `(n_atoms, features)` | aux · eval · MD |
| `a0_ratios` | Polarizability-volume ratios (`so3lr-tauri-s/m/l`) | `(n_atoms,)` | aux · eval † |
| `hirshfeld_ratios` | Hirshfeld volume ratios (legacy `so3lr`) | `(n_atoms,)` | aux · eval † |
| `c6_ratios` | Per-atom C6-coefficient ratios (`so3lr-tauri-s/m/l`) | `(n_atoms,)` | aux · eval † |
| `dipole_vec` | Molecular dipole vector | `(1, 3)` in aux | aux · eval † |
| `energy` | Total energy | `(n_atoms,)` in aux; per structure in eval | aux · eval † |
| `electrostatic_energy_kspace` | Reciprocal-space electrostatics (needs a [k-space method](electrostatics.md)) | `(n_atoms,)` | aux |

† **Label targets**, which `so3lr eval` scores against your dataset (as are
`forces`). Missing labels give `NaN` metrics.

With `has_aux=True`, `aux` always contains the model's head outputs (`energy`,
`dipole_vec`, `a0_ratios`, `c6_ratios`). The `observables` argument adds the
other internal quantities (`nn_energy`, `electrostatic_energy`, `x`, …).

### Accepted aliases

`so3lr eval --targets` also accepts these spellings:

| You write | Resolves to |
|---|---|
| `charges` | `partial_charges` |
| `hirsh_ratios` | `hirshfeld_ratios` |

## Names differ between model generations

This is the most common source of `KeyError`s.

| Quantity | `so3lr-tauri-s/m/l` (current) | `so3lr` (legacy v1) |
|---|---|---|
| Short-range repulsion | `nlh_repulsion`: NLH (also under `zbl_repulsion`) | `zbl_repulsion`: ZBL |
| Polarizability / volume ratio | `a0_ratios` | `hirshfeld_ratios` |
| C6 ratios | `c6_ratios` | none (derived from `hirshfeld_ratios`) |

Use `a0_ratios` unless you selected `model="so3lr"`; the default model's `aux`
has no `hirshfeld_ratios` key. [`so3lr eval`](../usage/evaluation.md) accepts
either ratio name on `--targets`.

## From the CLI

To save observables for a dataset, add them to `--targets` in `so3lr eval`
(see [Observable targets](../usage/evaluation.md#observable-targets)). They go to
the extxyz output, except `x`, which goes to a separate file (see
[Atom embeddings](../usage/evaluation.md#atom-embeddings)).

## During MD

Observables can also be recorded alongside the MD trajectory, with the
`--observable` flag or in a settings file:

```shell
so3lr nvt --input geometry.xyz --temperature 300 \
    --observable nn_energy --observable partial_charges
```

```yaml title="settings.yaml"
observables: [nn_energy, partial_charges, electrostatic_energy]
```

Values are written at each trajectory save step (every `md_steps`). Recording
observables runs the [neighbour-list engine](../usage/md.md#md-engines) instead
of the flash engine.

`--observable` takes only the names marked MD in the table above. The
settings-file `observables:` list also records head outputs such as
`dipole_vec`. Otherwise, extract the ratio heads and dipoles from the
trajectory afterwards with [`so3lr eval`](../usage/evaluation.md).

## Checking the decomposition

The components sum to the total energy, which is a useful sanity check for any
custom analysis:

```python
total = (np.sum(aux["nn_energy"])
         + np.sum(aux["electrostatic_energy"])
         + np.sum(aux["dispersion_energy"])
         + np.sum(aux["nlh_repulsion"]))

assert np.isclose(total, energy, atol=1e-6)
```

With k-space electrostatics on, add `electrostatic_energy_kspace` as well.
Use `dtype=np.float64`; in `float32` the residual can look like a real
discrepancy.

## Related

- [Gradients and Hessians](derivatives.md): derivatives of these observables
- [Long-range electrostatics](electrostatics.md): splitting real and reciprocal space
