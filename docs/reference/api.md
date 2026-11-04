# API reference

The public Python API is small. Everything below is importable from `so3lr`.

```python
from so3lr import So3lrCalculator, So3lrPotential, to_jax_md, Graph, So3lr
```

!!! warning "`float64` needs x64 enabled first"
    To use `dtype=np.float64` or float64 positions, enable 64-bit arrays
    before importing SO3LR:

    ```python
    from jax import config
    config.update("jax_enable_x64", True)
    ```

    The CLI does this for you.

## `So3lrCalculator`

The ASE calculator. See [ASE calculator](../usage/ase.md).

```python
So3lrCalculator(
    model=None,                                # bundled name or path; None -> so3lr-tauri-m-2606
    lr_cutoff=12.0,
    dispersion_energy_cutoff_lr_damping=2.0,
    calculate_stress=False,
    calculate_hessian=False,
    dtype=np.float32,
    has_aux=False,
    observables=None,
    output_atom_indices=None,
    kspace_electrostatics=None,                # 'ewald' | 'pme' | 'p3m'
    kspace_smearing=None,
    kspace_spacing=None,
    charge_embed_total_charge=None,            # None -> the system's total charge
    charge_equilibration_total_charge=None,    # None -> the system's total charge
    neighbor_backend="auto",                   # 'auto' | 'glp' | 'exact'
)
```

Results are stored in `calc.results`:

| Key | When |
|---|---|
| `energy`, `forces` | Always |
| `stress` | `calculate_stress=True` |
| `hessian` | `calculate_hessian=True` |
| `aux` | `has_aux=True`; see [Observables](../advanced/observables.md) |

`kspace_electrostatics` requires both `kspace_smearing` and `kspace_spacing`.

| Keyword | Meaning |
|---|---|
| `charge_embed_total_charge` | Overrides the total charge the network's charge embedding sees. |
| `charge_equilibration_total_charge` | Overrides the total charge that charge equilibration conserves. |
| `neighbor_backend` | `auto` uses explicit periodic images when the minimum image would be wrong; `glp` forces the minimum image, `exact` forces explicit images. The [`matscipy` extra](../installation.md#optional-extras) speeds up the explicit search. |

`calculate_stress`, `calculate_hessian` and observable `_grad`/`_jac` outputs
are mutually exclusive.

This name is an alias for `so3lr.ase_utils.make_ase_calculator`. The legacy
keyword `workdir=` takes precedence over `model=`. Unknown keywords are
forwarded into the long-range setup, not rejected, so a misspelling is not
reported.

## `So3lrPotential`

A pure JAX potential energy function. See
[Custom integrations](../advanced/integrations.md#the-potential-energy-function).

```python
So3lrPotential(
    model=None,                                # bundled name or path; None -> so3lr-tauri-m-2606
    lr_cutoff=12.0,
    dispersion_energy_cutoff_lr_damping=2.0,
    dtype=np.float32,
    coulomb_kspace_method="pme",               # 'ewald' | 'pme' | 'p3m'
    coulomb_kspace_interp_nodes=4,
    dispersion_kspace_method=None,             # None | 'ewald' | 'pme'
    dispersion_kspace_interp_nodes=4,
    charge_embed_total_charge=None,
    charge_equilibration_total_charge=None,
)
```

The returned object is called on a [`Graph`](#graph):

```python
potential = So3lrPotential(model="so3lr-tauri-m")

energies = potential(graph)                    # per-atom energies, shape (n_atoms,)
energies, aux = potential(graph, has_aux=True)
total_energy = energies.sum()
```

The gradient with respect to `graph.positions` is not the force; see
[Forces](../advanced/integrations.md#forces). This name is an alias for
`so3lr.potential.make_potential_fn`.

## `to_jax_md`

Connects a `So3lrPotential` to JAX-MD. See
[Custom integrations](../advanced/integrations.md#driving-jax-md-directly).

```text title="signature"
neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential,
    displacement_or_metric,
    box_size,                        # None -> treated as a very large box
    species=None,
    dr_threshold=0.,                 # > 0 is experimental
    capacity_multiplier=None,        # None -> the two buffer multipliers are used
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    minimum_cell_size_multiplier_lr=1.0,
    disable_cell_list=False,
    fractional_coordinates=True,
    total_charge=0.0,
    num_unpaired_electrons=0.0,
    precision=None,                  # None -> follow the positions dtype
    **neighbor_kwargs,               # forwarded to jax_md.partition.neighbor_list
)
```

`energy_fn(positions, neighbor=..., neighbor_lr=..., box=...)` returns the
total energy, or with `has_aux=True` the per-atom energies and the auxiliary
outputs.

!!! warning "`fractional_coordinates` defaults to `True`"
    Pass `fractional_coordinates=False` when you feed cartesian positions.

## `Graph`

The input type of `So3lrPotential`, a `collections.namedtuple`. Its fields are
listed in [Custom integrations](../advanced/integrations.md#the-graph-object).

!!! warning "`Graph.cell` holds lattice vectors as columns"
    `cell` is `atoms.cell.T`, or `None` in vacuum. Rows instead of columns give
    wrong k-space results on non-orthogonal cells, without an error.

## `So3lr`

A lower-level constructor (`so3lr.base_calculator.make_so3lr`) that returns
the raw model function, mapping an input dictionary to an output dictionary
(with `forces` unless `calculate_forces=False`). We recommend
`So3lrCalculator` or `So3lrPotential` instead.

```python
So3lr(
    model=None,                                # bundled name or path; None -> so3lr-tauri-m-2606
    lr_cutoff=12.0,
    dispersion_energy_cutoff_lr_damping=2.0,
    calculate_forces=True,
    output_intermediate_quantities=None,
    charge_embed_total_charge=None,
    charge_equilibration_total_charge=None,
    kspace_electrostatics=None,                # 'ewald' | 'pme' | 'p3m'
    kspace_interp_nodes=4,
    kspace_dispersion=None,                    # None | 'ewald' | 'pme'
    kspace_dispersion_interp_nodes=4,
)
```

Observable `_grad` / `_jac` suffixes work only through `So3lrCalculator`. See
[Gradients and Hessians](../advanced/derivatives.md).

## Model registry

`so3lr.model_registry` resolves names to bundled checkpoints.

```python
from so3lr.model_registry import (
    resolve_model, canonical_models, model_aliases, available_models,
    DEFAULT_MODEL,
)
```

| Function | Returns |
|---|---|
| `canonical_models()` | Tuple of canonical names, one per bundled directory. |
| `model_aliases()` | `dict` mapping each floating alias (`so3lr-tauri-s/m/l`) to its canonical name. |
| `available_models()` | Every accepted name, canonical first. |
| `resolve_model(name_or_path=None)` | A `ModelSpec`. `None` uses `DEFAULT_MODEL`. |
| `DEFAULT_MODEL` | The name used when nothing is specified: `"so3lr-tauri-m-2606"`. |

`resolve_model` raises `FileNotFoundError` when the argument is neither a
registered name nor an existing path.

### `ModelSpec`

A frozen dataclass with the following attributes:

| Attribute | Meaning |
|---|---|
| `dir` | Directory holding `params.pkl` and/or `checkpoints/`. |
| `family` | `"v2"` for the `so3lr-tauri-s/m/l` models, `"vanilla"` for the original `so3lr`, `"custom"` for user paths. |
| `head_index` | Theory-level head to read predictions from. |
| `from_file` | `True` when a `params.pkl` export is present. |

It also unpacks as the legacy `(dir, from_file, head_index)` triple, but
attribute access is preferred:

```python
spec = resolve_model("so3lr-tauri-m")
dir_, from_file, head = resolve_model("so3lr-tauri-m")   # still works
```

## Version

```python
import so3lr
so3lr.__version__
```

## Command-line entry point

`so3lr` on the command line maps to `so3lr.cli.so3lr_cli:main`. See
[Command line](../usage/cli.md).
