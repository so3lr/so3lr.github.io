# Gradients and Hessians

Beyond energies and forces, the ASE calculator can also compute second
derivatives and the derivatives of individual [observables](observables.md).
These are useful for vibrational analysis, transition-state searches and
method development.

## Hessians

Set `calculate_hessian=True` to get the second derivatives of the energy with
respect to the atomic positions:

```python
from jax import config

# Enable 64-bit arrays before importing SO3LR when using dtype=np.float64.
config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms
from so3lr import So3lrCalculator

atoms = Atoms("OH2", positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])

calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=100,          # gas phase
    dtype=np.float64,
    calculate_hessian=True,
)

atoms.calc = calc
atoms.get_potential_energy()

hessian = calc.results["hessian"]
# shape (n_atoms, 3, n_atoms, 3)
# hessian[i, a, j, b] = d²E / (dR_i^a dR_j^b)
```

For eigenvalue analysis, reshape to 2D:

```python
n = len(atoms)
h2d = hessian.reshape(n * 3, n * 3)
eigenvalues = np.linalg.eigvalsh(h2d)
```

!!! warning
    Use `dtype=np.float64` for Hessians; in `float32` the low-frequency
    eigenvalues are dominated by noise. `float64` also needs `jax_enable_x64`,
    as above, or the arrays silently stay `float32`.

### Subsystem Hessians

Full Hessians scale as O(N²) in memory. For large systems, restrict the
calculation to a contiguous block of atoms with `output_atom_indices`:

```python
# `atoms` here is a system of at least 10 atoms
calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=12.0,
    dtype=np.float64,
    calculate_hessian=True,
    output_atom_indices=list(range(10)),   # atoms 0-9, must be contiguous
)

atoms.calc = calc
atoms.get_potential_energy()

hessian = calc.results["hessian"]   # shape (10, 3, 10, 3)
```

This gives the subset × subset block of the full Hessian. The forces are
restricted to the same block, so `calc.results["forces"]` has shape `(10, 3)`.

## Observable gradients

To get the derivatives of an observable with respect to the atomic positions,
append `_grad` (for scalars) or `_jac` (for vectors) to its name and give
`output_atom_indices`:

```python
# same `atoms` as above
calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=100,
    dtype=np.float64,
    has_aux=True,
    observables=["nn_energy_grad", "partial_charges_jac"],
    output_atom_indices=[0, 1, 2],      # required
)

atoms.calc = calc
atoms.get_potential_energy()   # triggers the calculation

aux = calc.results["aux"]

# Scalar observable -> {atom_idx: (value, gradient)}
for idx in [0, 1, 2]:
    value, grad = aux["nn_energy_grad"][idx]
    # value: the atomic nn_energy of atom idx
    # grad:  shape (3,), d(sum over all atoms of nn_energy)/d(R_idx)
    print(f"atom {idx}: energy={value:.6f}, grad={grad}")

# Vector observable -> {atom_idx: (value, jacobian)}
for idx in [0, 1, 2]:
    value, jac = aux["partial_charges_jac"][idx]
    # value: scalar partial charge on atom idx
    # jac:   shape (n_atoms, 3), d(all charges)/d(R_idx)
    print(f"atom {idx}: charge={value:.4f}, jacobian={jac.shape}")
```

| Suffix | Use for | Returns, per atom in `output_atom_indices` |
|---|---|---|
| `_grad` | Scalar observables (energies) | Gradient of the observable summed over all atoms, shape `(3,)` |
| `_jac` | Vector observables (charges, ratio heads) | Jacobian of the whole per-atom array, shape `(n_atoms, 3)` |

`_grad` gives the derivative of the total, not of the atomic value it is paired
with: `nn_energy_grad` is minus the neural-network part of the force on that
atom.

In this mode the calculator computes only what you list, so
`calc.results["energy"]` is `None` and there are no forces. Run a separate
calculation for those.

## Restrictions

`calculate_hessian`, `calculate_stress` and observable gradients cannot be
combined; run separate calculations if you need more than one.

The `_grad` and `_jac` suffixes work only through `So3lrCalculator`, not
through `so3lr eval` or the lower-level `So3lr`.
