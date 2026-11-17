# Long-range electrostatics

For periodic systems, SO3LR can compute electrostatics in reciprocal space with
three methods from [jax-pme](https://github.com/lab-cosmo/jax-pme/) (a JAX port
of [torch-pme](https://github.com/lab-cosmo/torch-pme/)).

| Method | Scaling | Accuracy | Best for |
|---|---|---|---|
| `ewald` | O(N²) | Exact | Small systems (< ~1000 atoms), reference values |
| `pme` | O(N log N) | High | Large systems, production MD |
| `p3m` | O(N log N) | Higher | When accuracy is critical at a given grid size |

## When you need it

Use k-space electrostatics for periodic systems (bulk liquids, crystals,
interfaces), especially charged ones. The command line does this by default:
`so3lr nvt`, `npt`, `nve`, `opt`, `eval` and `so3lr --settings` use PME for
electrostatics and dispersion on a structure periodic along all three axes (see
[Long-range defaults](../usage/cli.md#long-range-defaults)). The ASE calculator
and `tune-ewald` use k-space only when asked. For gas-phase and isolated
molecules, set `lr_cutoff=100` instead, so that nothing is truncated.

K-space needs a cell that is periodic along all three axes. Slabs and wires
(`pbc=[True, True, False]`) are rejected.

## From the command line

A periodic box needs no flag: PME is on, with a 10 Å cutoff, 2 Å smearing and
a 1.55 Å mesh. To choose the method or the parameters:

```shell
so3lr nvt --input water_box.xyz --temperature 300 --dt 0.5 \
    --md-cycles 100 --md-steps 100 \
    --kspace-electrostatics pme \
    --kspace-smearing 2.0 \
    --kspace-spacing 1.0
```

| Flag | Default | Meaning |
|---|---|---|
| `--kspace-electrostatics` | `auto` | `auto` (PME on a fully periodic structure), `none`, `ewald`, `pme` or `p3m` |
| `--kspace-smearing` | `2.0` | Smearing width, Å |
| `--kspace-spacing` | `1.55` | Grid spacing (PME/P3M) or wavelength (Ewald), Å; 1 below a 10 Å cutoff, 0.775 × a given smearing |
| `--kspace-interp-nodes` | `4` | Interpolation nodes for PME/P3M |
| `--kspace-npt-cycles` | every 100 cycles | Rebuild the grid for the current box every N cycles under NPT |

The MD subcommands and `opt` take all five, and `so3lr eval` the first four (see
[below](#from-so3lr-eval)). `tune-ewald` takes all five, without `auto` and
`none`, and keeps 4 Å smearing and 2 Å spacing as its defaults.
`--kspace-electrostatics none --kspace-dispersion none` keeps both terms in real
space.

The same options can be set in a settings file. A key left out is `auto`;
`none` or an explicit `null` turns the term off:

```yaml title="settings.yaml"
kspace_electrostatics: pme      # auto / none / ewald / pme / p3m
kspace_smearing: 2.0
kspace_spacing: 1.0
kspace_interp_nodes: 4
```

Under NPT the grid is rebuilt for the current box every `kspace_npt_cycles`
cycles (100 when unset, never with `kspace_npt_cycles: null`).

### Box size, cell shape and engine

- **Cell shape.** MD runs orthorhombic cells only. The
  [ASE calculator](#from-the-ase-calculator) takes any cell shape.
- **Box size.** The [flash engine](../usage/md.md#when-flash-runs) takes
  small boxes. On the neighbour-list engine every box length must be at least
  twice `lr_cutoff` (20 Å at the 10 Å PME default). Unless you set
  `--lr-cutoff`, a smaller periodic box gets a cutoff of 0.45 × its smallest
  height and a 1 Å mesh, with both terms in k-space. When you lower
  `--lr-cutoff` yourself, keep dispersion in k-space too, because `lr_cutoff`
  truncates the real-space dispersion. An `lr_cutoff` below 10 Å is accepted
  only with k-space electrostatics on. See [Box size](../usage/md.md#box-size).
- **Engine.** The flash engine runs PME (electrostatics and dispersion) with
  4 interpolation nodes. Ewald, P3M, other node counts, an explicit
  `kspace_npt_cycles` and the dispersion tail correction run on the
  neighbour-list engine. See
  [When flash runs](../usage/md.md#when-flash-runs).
- **Legacy model.** K-space has not been tested with the original `so3lr`
  model, which was trained with real-space electrostatics.

## From the ASE calculator

In Python, the parameter is `kspace_electrostatics`, as on the command line.
It needs `kspace_smearing` and `kspace_spacing`, and the calculator builds the
k-grid from the cell:

```python
from jax import config

# Enable 64-bit arrays before importing SO3LR; precision="float64" needs them.
config.update("jax_enable_x64", True)

import numpy as np
from ase.io import read
from so3lr import So3lrCalculator

atoms = read("water_box.xyz")

atoms.calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=12.0,
    precision="float64",
    kspace_electrostatics="pme",     # 'ewald', 'pme' or 'p3m'
    kspace_smearing=4.0,
    kspace_spacing=0.5,
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The k-space arguments are ignored for non-periodic systems. The calculator
handles triclinic and small cells (see
[Periodic systems](../usage/ase.md#periodic-systems)). PME/P3M interpolate with
4 nodes, which you can change with `coulomb_kspace_interp_nodes=`. K-space
dispersion is not available through the calculator; use MD or `so3lr eval`.

### Supplying a grid yourself

For full control, put the grid in `atoms.info`. It overrides the grid the
calculator would build, and `coulomb_kspace_method` selects the solver:

```python
from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

cell = np.array(atoms.get_cell())   # lattice vectors as rows, as jax-pme expects
smearing = 4.0

# PME / P3M use a mesh grid
k_grid = get_kgrid_mesh(cell, np.array([smearing / 8]))

# Ewald uses an ewald grid instead
# k_grid = get_kgrid_ewald(cell, np.array([smearing / 2]))

atoms.info["k_grid"] = k_grid
atoms.info["k_smearing"] = np.array([smearing])

atoms.calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=12.0,
    precision="float64",
    coulomb_kspace_method="pme",   # must match the grid you built above
)
```

Set `k_grid` and `k_smearing` together. Nothing checks that the grid
constructor and `coulomb_kspace_method` agree.

### The two spellings

| Argument | Layer | Effect |
|---|---|---|
| `kspace_electrostatics` | calculator | Builds the k-grid from the cell and selects the solver. Recommended. |
| `coulomb_kspace_method` | model | Selects the solver only (default `pme`), for a hand-built `atoms.info["k_grid"]`. |

[`So3lrPotential`](integrations.md#the-potential-energy-function) takes the
model-level spelling.

## Splitting the energy

To get the real-space and k-space parts of the electrostatic energy
separately, request them as observables:

```python
calc = So3lrCalculator(
    model="so3lr-tauri-m",
    lr_cutoff=12.0,
    precision="float64",
    kspace_electrostatics="pme",
    kspace_smearing=4.0,
    kspace_spacing=0.5,
    has_aux=True,
    observables=[
        "electrostatic_energy",         # short-range (real-space) part
        "electrostatic_energy_kspace",  # long-range (k-space) part
        "partial_charges",
    ],
)

atoms.calc = calc
atoms.get_potential_energy()

aux = calc.results["aux"]
print("real space:", np.sum(aux["electrostatic_energy"]))
print("k-space:   ", np.sum(aux["electrostatic_energy_kspace"]))
print("net charge:", np.sum(aux["partial_charges"]))
```

The electrostatic contribution to the total energy is the sum of both parts.

## From `so3lr eval`

`so3lr eval` takes `--kspace-electrostatics`, `--kspace-smearing`,
`--kspace-spacing` and `--kspace-interp-nodes`, plus the k-space dispersion
flags: `--kspace-dispersion` (`auto`, `none`, `ewald` or `pme`),
`--kspace-dispersion-smearing` (default 2.0 Å), `--kspace-dispersion-spacing`
(default 3.0 Å) and `--kspace-dispersion-interp-nodes` (default 4). Both terms
default to PME when every structure in the file is fully periodic, with a 10 Å
cutoff that is not shortened for small cells.

```shell
so3lr eval --datafile crystals.extxyz --lr-cutoff 12 \
    --kspace-electrostatics pme --kspace-smearing 4.0 --kspace-spacing 0.5 \
    --save-to crystals_pred.extxyz
```

With k-space asked for explicitly, every structure in the file must be fully
periodic. See [Dataset evaluation](../usage/evaluation.md).

## Choosing parameters by hand

`smearing` sets the real-space/k-space split:

- larger smearing → more work in k-space, less in real space;
- typical values: 3.0–5.0 Å.

`spacing` sets the grid density:

- smaller spacing → finer grid, higher accuracy, more computation;
- PME/P3M: `spacing ≈ smearing / 8` is a reasonable start;
- Ewald: `spacing ≈ smearing / 2` is typical.

`spacing` is not the literal mesh spacing. For Ewald, each lattice direction of
length *L* gets ⌈*L* / spacing⌉ wave vectors. For PME/P3M, the mesh has the
next power of two at or above 2⌈*L* / spacing⌉ + 1 points along that
direction, so the cost doubles whenever a change in box or spacing crosses a
power of two.

## Automatic tuning

`so3lr tune-ewald` grid-searches the real-space cutoff, smearing and grid
resolution. Of the parameter sets that reach a target force accuracy, it
recommends the most accurate one that is about as fast as the fastest (see
below):

```shell
so3lr tune-ewald --input water_box.xyz \
    --kspace-electrostatics pme \
    --lr-cutoff 12.0 \
    --precision float64
```

| Flag | Meaning |
|---|---|
| `--kspace-electrostatics` | Method to tune: `ewald`, `pme` or `p3m` |
| `--lr-cutoff` | Upper bound of the cutoff search, Å |
| `--precision` | Use `float64` for meaningful error bounds |
| `--engine` | `auto` (default), `flash` or `standard`; see below |
| `--settings` | Optional YAML with the search ranges below |

The other model flags are shared with the MD commands. The input must have a
cell, and every cell height must be at least twice `--lr-cutoff`.
Per-candidate errors and timings go to the log file (`<input>_tune-kspace.log`,
or `<settings>.log` with `--settings`), and the command prints a ready-to-run
MD command and the equivalent settings keys.

PME tuning runs on the [flash engine](../usage/md.md#when-flash-runs) whenever
flash takes the structure. The partial charges come from flash, and each
candidate is timed with flash force calls, one engine per cutoff and mesh shape.
Changing the smearing does not change the work. With no per-edge arrays, systems
of tens of thousands of atoms fit on one GPU. Otherwise, and always for Ewald,
P3M and the `cutoff` modes, the tuning runs on the neighbour-list potential.
`--engine standard` forces the neighbour lists. `--engine flash` forces flash and
fails where flash cannot run. The log says which engine ran and why.

The real-space error is measured on the pairs of up to 2048 sampled atoms (all
of them in smaller systems). The k-space error uses the analytic estimates,
with the mesh jax-pme builds (padded to a power of two). Of the candidates
within `accuracy`, the recommendation is the most accurate one whose timing is
within 3 % of the fastest. On flash the mesh costs little next to the model, so
timings of neighbouring candidates differ mostly by noise.

The search ranges are set in a settings file passed with `--settings` (also
listed in the [settings reference](../usage/settings.md#tune-ewald-search-ranges)):

| Key | Default | Meaning |
|---|---|---|
| `accuracy` | `1.0e-3` | Target RMS force error, eV/Å |
| `cutoff_lo`, `cutoff_hi`, `cutoff_it` | `7.0` (`10.0` in the `cutoff` modes), `lr_cutoff`, `2.0` | Real-space cutoff range and step, Å |
| `smear_lo`, `smear_hi`, `smear_it` | `2.0`, `6.0`, `2.0` | Smearing range and step, Å |
| `ns_lo`, `ns_hi` | `4`, `8` | Ewald: candidate spacings *L*/*n* for *n* in this range, where *L* is the shortest cell vector |
| `mesh_lo`, `mesh_hi` | `5`, `7` | PME/P3M: candidate spacings that give 2ⁿ mesh points along the shortest cell vector, for *n* in this range |
| `nodes_lo`, `nodes_hi` | `4`, `4` | Interpolation nodes: PME takes only 4, P3M takes 3–5 |

A settings file can also set `kspace_electrostatics: cutoff`, which tunes only
the real-space `lr_cutoff`, or `kspace_dispersion: cutoff`, which tunes
`lr_cutoff` and `dispersion_damping` for the dispersion. The MD command and
settings these modes print turn k-space off (`--kspace-electrostatics none
--kspace-dispersion none`), since a periodic structure would otherwise get PME.

## K-space dispersion

The dispersion term can be treated the same way, which helps in dense periodic
systems where the r⁻⁶ tail is still significant at the cutoff. The command line
uses PME for it on fully periodic structures by default. The MD commands, `opt`
and `eval` take `--kspace-dispersion` (`auto`, `none`, `ewald` or `pme`),
`--kspace-dispersion-smearing`, `--kspace-dispersion-spacing` and
`--kspace-dispersion-interp-nodes`; the settings keys are:

```yaml
kspace_dispersion: pme                  # auto | none | ewald | pme
kspace_dispersion_smearing: 2.0         # Å, default 2.0
kspace_dispersion_spacing: 3.0          # Å, default 3.0
kspace_dispersion_interp_nodes: 4       # default 4
```

A cheaper alternative under NPT is the analytic mean-field tail correction,
which estimates the dispersion energy beyond the cutoff:

```yaml
dispersion_tail_correction: true
```

Other ensembles ignore the key. It works with the `so3lr-tauri-s/m/l` models,
not with the legacy `so3lr` model, and runs on the neighbour-list engine.
