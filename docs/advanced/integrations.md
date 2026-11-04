# Custom integrations

This page is for embedding SO3LR into your own simulation code, through the
potential energy function and the JAX-MD interface. For standard simulations
use the [CLI](../usage/cli.md), which also runs the faster
[flash engine](../usage/md.md#md-engines), and for Python-level work use the
[ASE calculator](../usage/ase.md).

## Which interface should I use?

| You want to… | Use |
|---|---|
| Run standard MD, optimization or evaluation | [The CLI](../usage/cli.md) |
| Use ASE tooling (optimizers, NEB, vibrations) | [`So3lrCalculator`](../usage/ase.md) |
| Write a custom JAX-MD simulation | [`to_jax_md`](#driving-jax-md-directly) |
| Embed SO3LR in a non-JAX-MD code | [`So3lrPotential`](#the-potential-energy-function) |

## The potential energy function

`So3lrPotential` is a pure `jax` function which takes a `Graph` as input and
returns the per-atom potential energies with shape `(n_atoms,)`, so you sum
them to get the total energy. It works with `jax.jit`, `jax.vmap` and
`jax.grad`, and has no JAX-MD dependency.

```python
from so3lr import So3lrPotential, Graph

# `Graph` is built from your own system. See "The Graph object" below.
graph = Graph(...)

so3lr_potential = So3lrPotential(model="so3lr-tauri-m")
energy = so3lr_potential(graph).sum()
```

The full signature is in the [API reference](../reference/api.md#so3lrpotential).

### The `Graph` object

The `Graph` object is a `collections.namedtuple` which abstracts the molecule
as a graph. Its fields are:

| Field | Contents |
|---|---|
| `nodes` | Atomic numbers, `(n_atoms,)` |
| `positions` | Cartesian positions, Å, `(n_atoms, 3)` |
| `centers`, `others`, `edges` | Short-range pairs within the model's cutoff, **both directions**, and their displacement vectors `(n_pairs, 3)` |
| `idx_i_lr`, `idx_j_lr`, `edges_lr` | Long-range pairs within `lr_cutoff`, **each pair once**, and their displacement vectors |
| `total_charge`, `num_unpaired_electrons` | Shape `(1,)` |
| `cell` | Lattice vectors as **columns** (`atoms.cell.T`), or `None` in vacuum |
| `theory_mask` | The theory-level head to read. Required; use `jnp.eye(16)[5][None]`. |
| `k_grid`, `k_smearing`, `k_grid_disp`, `k_smearing_disp` | K-space grids and smearings; optional |
| `residue_segments`, `residue_charge` | Per-fragment charge constraints; optional |
| `mask` | Not read by `So3lrPotential` |

Listing a long-range pair in both directions double-counts it. Unused index
slots may be padded with `n_atoms` in both rows, as jax-md does.

To integrate SO3LR into an existing code, write a function that turns your
input structure into a `Graph`:

```python
def system_to_graph(my_system) -> Graph:
    """Translate your internal representation into a SO3LR Graph."""
    ...
```

### Forces

The network reads the displacement vectors in `edges` and `edges_lr`, so the
gradient with respect to `graph.positions` is **not** the force. Rebuild the
displacements from the positions inside the function you differentiate:

```python
import jax

def with_positions(R, template):
    """The template's pairs, with displacements recomputed from positions R."""
    return template._replace(
        positions=R,
        edges=R[template.centers] - R[template.others],
        edges_lr=R[template.idx_i_lr] - R[template.idx_j_lr],
    )

@jax.jit
def energy_and_forces(R, template):
    def total_energy(positions):
        return so3lr_potential(with_positions(positions, template)).sum()

    energy, grad = jax.value_and_grad(total_energy)(R)
    return energy, -grad
```

This is the free-space form; in a periodic box, apply the minimum-image
convention to the differences. [`to_jax_md`](#driving-jax-md-directly) does
this for you.

### Auxiliary outputs

Like the ASE calculator, the potential can return intermediate quantities:

```python
energies, aux = so3lr_potential(graph, has_aux=True)
```

See [Observables](observables.md) for what `aux` contains.

## Driving JAX-MD directly

[JAX-MD](https://github.com/jax-md/jax-md) is a molecular dynamics library
optimized for GPUs, and `to_jax_md` connects a `So3lrPotential` to it.

### A minimal gas-phase example

```python
import jax
import jax.numpy as jnp
import numpy as np

from ase import Atoms
from jax_md import space, quantity

from so3lr import to_jax_md, So3lrPotential

atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

positions = jnp.array(atoms.get_positions())
atomic_numbers = jnp.array(atoms.get_atomic_numbers())

# Assume there is no box (free space)
box = None
displacement, shift = space.free()

neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=So3lrPotential(model="so3lr-tauri-m", lr_cutoff=100.0),  # gas phase
    displacement_or_metric=displacement,
    box_size=box,
    species=atomic_numbers,
    capacity_multiplier=1.25,
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    disable_cell_list=True,
    fractional_coordinates=False,
)

energy_fn = jax.jit(energy_fn)
force_fn = jax.jit(quantity.force(energy_fn))

# Allocate the short- and long-range neighbor lists
nbrs = neighbor_fn.allocate(positions, box=box)
nbrs_lr = neighbor_fn_lr.allocate(positions, box=box)

energy = energy_fn(positions, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
forces = force_fn(positions, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)

print("Energy =", np.array(energy))
print("Forces =", np.array(forces))
```

### `to_jax_md` arguments

| Argument | Meaning |
|---|---|
| `potential` | A `So3lrPotential` instance. |
| `displacement_or_metric` | From `jax_md.space.free()` or `space.periodic_general(...)`. |
| `box_size` | `None` for free space, otherwise the box. |
| `species` | Atomic numbers. |
| `capacity_multiplier` | One capacity multiplier for both lists; `None` (default) uses each list's buffer multiplier. |
| `buffer_size_multiplier_sr` / `_lr` | Short- and long-range buffer headroom (default 1.25). |
| `minimum_cell_size_multiplier_sr` / `_lr` | Cell-list cell size multipliers (default 1.0). |
| `disable_cell_list` | `True` for small or non-periodic systems. |
| `fractional_coordinates` | Whether positions are fractional. Defaults to **`True`** and must match your `space` choice. |
| `total_charge`, `num_unpaired_electrons` | Net charge (e) and spin of the system (default 0). |
| `precision` | Dtype of the auxiliary arrays. `None` follows the positions. |

Anything else is forwarded to `jax_md.partition.neighbor_list`.

`to_jax_md` returns a short-range neighbor function, a long-range neighbor
function, and an energy function taking
`(positions, neighbor=..., neighbor_lr=..., box=...)`. The energy function
returns the total energy, or with `has_aux=True` the per-atom energies and the
[auxiliary outputs](observables.md).

### Periodic systems

For a periodic box, use `space.periodic_general` and pass the box through. A
jax-md box holds the lattice vectors as **columns**, so transpose the ASE cell:

```python
from jax_md import space

box = jnp.array(atoms.get_cell().array.T)          # lattice vectors as columns
positions = jnp.array(atoms.get_scaled_positions())  # fractional, to match below
displacement, shift = space.periodic_general(
    box, fractional_coordinates=True
)

neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=So3lrPotential(model="so3lr-tauri-m"),
    displacement_or_metric=displacement,
    box_size=box,
    species=atomic_numbers,
    capacity_multiplier=1.25,
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    disable_cell_list=False,
    fractional_coordinates=True,
)
```

Every box length must be at least twice `lr_cutoff`. In a smaller box,
interactions with other periodic images are dropped without an error, so check
it yourself or use the [ASE calculator](../usage/ase.md).

For k-space electrostatics, also pass `k_grid=...` and `k_smearing=...` to the
energy function. See [Long-range electrostatics](electrostatics.md).

!!! warning "Neighbor lists overflow silently"
    Neighbor lists are allocated with a fixed capacity. During a simulation you
    must check `did_buffer_overflow` and reallocate:

    ```python
    nbrs = neighbor_fn.update(positions, nbrs)
    if nbrs.did_buffer_overflow:
        nbrs = neighbor_fn.allocate(positions, box=box)
    ```

    Otherwise interactions are dropped with no error and the forces are wrong.
    The CLI handles this for you.

### Units and other traps

- **Units.** JAX-MD's metal units have their own time unit. Multiply a
  timestep in picoseconds by `units["time"]` and a temperature in kelvin by
  `units["temperature"]`, with `units = jax_md.units.metal_unit_system()`. A
  raw `dt=0.0005` runs steps about 98× too short, with no visible error.
- **Minimizer settings.** `jax_md.minimize.fire_descent`'s defaults are too
  aggressive for SO3LR. The CLI uses `dt_start=0.05`, `dt_max=0.1`, `n_min=2`.
- **One dtype throughout.** Keep positions, masses and kT in the same
  precision, or the neighbour-list update inside a compiled loop can fail.

## Worked examples

The [example notebooks](../examples.md) contain complete NVT and NPT loops with
neighbor-list handling, trajectory writing and checkpointing:

- `examples/nvt_jaxmd.ipynb`
- `examples/npt_jaxmd.ipynb`
- `examples/production_run.ipynb`
