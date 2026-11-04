# ASE calculator

`So3lrCalculator` is an Atomic Simulation Environment
([ASE](https://wiki.fysik.dtu.dk/ase/)) calculator that predicts energies and
forces using SO3LR. It works in any ASE-based workflow, such as optimizers,
vibrational analysis, NEB and `ase.md`.

## Basic use

```python
from jax import config

# Needed for dtype=np.float64; must run before SO3LR is imported.
config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms
from so3lr import So3lrCalculator

atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
atoms.info["charge"] = 0.0

atoms.calc = So3lrCalculator(
    model="so3lr-tauri-m-2606",   # bundled name or path; this is the default
    lr_cutoff=100,          # 100 for gas phase; ~12 for periodic systems
    calculate_stress=False,
    dtype=np.float64,
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Å
```

!!! warning "`float64` needs `jax_enable_x64`"
    Without the `config.update` line above, JAX truncates `float64` arrays to
    `float32` with only a warning. The CLI sets it for you.

Units are eV and Å. The legacy `so3lr` model was not trained on energies, so
only its relative energies are meaningful.

## Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `model` | `so3lr-tauri-m-2606` | Bundled name or path to a custom model. See [Models](../models.md). |
| `lr_cutoff` | `12.0` | Long-range cutoff, Å. Use `100` for isolated molecules. |
| `dispersion_energy_cutoff_lr_damping` | `2.0` | Dispersion switch-off width, Å. |
| `dtype` | `np.float32` | `np.float32` or `np.float64`. |
| `calculate_stress` | `False` | Also return the stress tensor (periodic systems only). |
| `calculate_hessian` | `False` | Return the Hessian. See [Gradients and Hessians](../advanced/derivatives.md). |
| `has_aux` | `False` | Expose intermediate quantities in `calc.results["aux"]`. |
| `observables` | `None` | Which intermediates to expose. See [Observables](../advanced/observables.md). |
| `output_atom_indices` | `None` | Restrict Hessian/gradient output to a subset of atoms. |
| `kspace_electrostatics` | `None` | `ewald` / `pme` / `p3m`. See [Long-range electrostatics](../advanced/electrostatics.md). |
| `kspace_smearing` | `None` | Ewald smearing width, Å. Required with `kspace_electrostatics`. |
| `kspace_spacing` | `None` | K-grid spacing, Å. Required with `kspace_electrostatics`. |
| `charge_embed_total_charge` | `None` | Charge seen only by the charge embedding; `None` follows the structure. |
| `charge_equilibration_total_charge` | `None` | Charge that charge equilibration conserves to; `None` follows the structure. |
| `neighbor_backend` | `"auto"` | Periodic neighbour list: `"auto"`, `"glp"` or `"exact"`. See [below](#small-and-skewed-cells). |

`calculate_stress`, `calculate_hessian` and observable gradients are mutually
exclusive.

## Total charge and spin

Set the total charge and spin multiplicity on the `Atoms` object. When they
are absent, the structure is treated as a neutral singlet.

```python
atoms.info["charge"] = -1.0
atoms.info["multiplicity"] = 2      # 2S + 1
```

On the command line, the equivalent flags are `--total-charge` and `--spin`.

## Periodic systems

```python
from ase.io import read

atoms = read("water_box.xyz")           # carries a cell and pbc
atoms.calc = So3lrCalculator(
    lr_cutoff=12.0,
    calculate_stress=True,
    dtype=np.float64,
)

energy = atoms.get_potential_energy()
stress = atoms.get_stress()
```

For a periodic system with a net charge, use
[k-space electrostatics](../advanced/electrostatics.md) rather than raising
`lr_cutoff`. Periodicity must be all or nothing: mixed `pbc` flags (slabs,
wires) are rejected.

### Small and skewed cells

The calculator accepts any cell shape, including cells shorter than twice the
cutoff. With `neighbor_backend="auto"`, it lists every periodic image
explicitly when a minimum-image list would be wrong (skewed or thin cells), and
uses the faster minimum-image list otherwise. `"exact"` and `"glp"` force one or
the other.

The explicit list is much faster with
[matscipy](https://github.com/libAtoms/matscipy) installed:

```shell
uv sync --locked --extra matscipy     # or, from the checkout: pip install ".[matscipy]"
```

## Geometry optimization with ASE

```python
from ase.optimize import BFGS

atoms.calc = So3lrCalculator(model="so3lr-tauri-m", lr_cutoff=100)
BFGS(atoms).run(fmax=0.01)
```

The [`so3lr opt`](cli.md#geometry-optimization-opt) CLI command is faster for
large systems.

## Choosing precision

`float32` is roughly twice as fast as `float64` and is fine for MD. Use `float64`
(with `jax_enable_x64`, see [above](#basic-use)) for tight force convergence,
clean Hessians, or [`tune-ewald`](../advanced/electrostatics.md#automatic-tuning)
accuracy estimates.

## Related

- [Observables](../advanced/observables.md): energy components, partial charges, embeddings
- [Gradients and Hessians](../advanced/derivatives.md): second derivatives and observable Jacobians
- [Custom integrations](../advanced/integrations.md): driving JAX-MD yourself, or embedding SO3LR in your own code
