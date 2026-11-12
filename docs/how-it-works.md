# How SO3LR works

SO3LR couples a neural network to explicit physics. The total energy is a sum
of four terms:

<p class="so3lr-equation">E = E<sub>rep</sub> + E<sub>SO3k</sub> + E<sub>elec</sub> + E<sub>disp</sub></p>

The SO3krates network sees each atom's neighbourhood and predicts the
semi-local energy, together with per-atom quantities that set up the
long-range terms. Repulsion is fixed and has no trainable parameters. The
boxes link to the pages where each term is used.

<figure class="so3lr-diagram">
<svg viewBox="0 0 850 380" role="img" aria-labelledby="so3lr-diagram-title so3lr-diagram-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="so3lr-diagram-title">How SO3LR computes the energy</title>
  <desc id="so3lr-diagram-desc">Positions, elements, total charge and spin multiplicity go into the SO3krates network, which predicts atomic energies, partial charges, and polarizability and C6 ratios. These feed the semi-local energy, electrostatics and dispersion. Short-range repulsion uses positions and elements directly. The four terms sum to the total energy; forces are its negative gradient.</desc>
  <defs>
    <marker id="so3lr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="so3lr-diagram__head"/>
    </marker>
  </defs>
  <g class="so3lr-diagram__box">
    <rect x="10" y="130" width="140" height="140" rx="8"/>
    <text x="80" y="160" class="so3lr-diagram__title">Input</text>
    <text x="80" y="188">positions R</text>
    <text x="80" y="210">elements Z</text>
    <text x="80" y="232">total charge Q</text>
    <text x="80" y="254">spin multiplicity</text>
  </g>
  <a href="../models/">
    <g class="so3lr-diagram__box so3lr-diagram__box--nn">
      <rect x="200" y="150" width="180" height="100" rx="8"/>
      <text x="290" y="182" class="so3lr-diagram__title">SO3krates</text>
      <text x="290" y="208">message passing</text>
      <text x="290" y="230">within 4.5–5 Å</text>
    </g>
  </a>
  <a href="../advanced/observables/#available-observables">
    <g class="so3lr-diagram__box">
      <rect x="440" y="14" width="230" height="72" rx="8"/>
      <text x="555" y="42" class="so3lr-diagram__title">Short-range repulsion</text>
      <text x="555" y="66">NLH pair potentials, fixed</text>
    </g>
  </a>
  <a href="../advanced/observables/">
    <g class="so3lr-diagram__box">
      <rect x="440" y="106" width="230" height="72" rx="8"/>
      <text x="555" y="134" class="so3lr-diagram__title">Semi-local energy</text>
      <text x="555" y="158">sum of atomic energies Eᵢ</text>
    </g>
  </a>
  <a href="../advanced/electrostatics/">
    <g class="so3lr-diagram__box">
      <rect x="440" y="198" width="230" height="72" rx="8"/>
      <text x="555" y="222" class="so3lr-diagram__title">Electrostatics</text>
      <text x="555" y="242">damped Coulomb on charges qᵢ</text>
      <text x="555" y="260">optional k-space, 3D-periodic</text>
    </g>
  </a>
  <a href="../advanced/electrostatics/#k-space-dispersion">
    <g class="so3lr-diagram__box">
      <rect x="440" y="290" width="230" height="72" rx="8"/>
      <text x="555" y="314" class="so3lr-diagram__title">Dispersion</text>
      <text x="555" y="334">C₆, C₈, C₁₀ from α and C₆ ratios</text>
      <text x="555" y="352">optional k-space C₆, 3D-periodic</text>
    </g>
  </a>
  <g class="so3lr-diagram__arrows">
    <path d="M150,200 H196" marker-end="url(#so3lr-arrow)"/>
    <path d="M80,130 V50 H436" marker-end="url(#so3lr-arrow)"/>
    <path d="M380,180 L436,144" marker-end="url(#so3lr-arrow)"/>
    <path d="M380,205 L436,232" marker-end="url(#so3lr-arrow)"/>
    <path d="M380,230 L436,322" marker-end="url(#so3lr-arrow)"/>
    <path d="M670,50 L700,175" marker-end="url(#so3lr-arrow)"/>
    <path d="M670,142 L694,183" marker-end="url(#so3lr-arrow)"/>
    <path d="M670,234 L694,197" marker-end="url(#so3lr-arrow)"/>
    <path d="M670,326 L700,205" marker-end="url(#so3lr-arrow)"/>
  </g>
  <g class="so3lr-diagram__labels">
    <text x="260" y="44">R, Z</text>
    <text x="398" y="146">Eᵢ</text>
    <text x="410" y="204">qᵢ</text>
    <text x="372" y="292">α, C₆ ratios</text>
  </g>
  <g class="so3lr-diagram__box so3lr-diagram__box--sum">
    <circle cx="715" cy="190" r="22"/>
    <text x="715" y="198" class="so3lr-diagram__title">Σ</text>
  </g>
  <text x="748" y="186" class="so3lr-diagram__title so3lr-diagram__free">energy E</text>
  <text x="748" y="206" class="so3lr-diagram__free">forces −∇E</text>
</svg>
</figure>

| Term | Computed from | Range | Observable |
|---|---|---|---|
| Short-range repulsion | Nordlund–Lehtola–Hobler pair potentials, fixed | below about 1.5 Å | `nlh_repulsion` |
| Semi-local energy | SO3krates atomic energies | 4.5 Å per message-passing layer (5.0 Å for `-l`), 2–3 layers | `nn_energy` |
| Electrostatics | predicted partial charges, damped Coulomb | `lr_cutoff`; optionally every pair with k-space | `electrostatic_energy`, plus `electrostatic_energy_kspace` |
| Dispersion | predicted polarizability and C₆ ratios | `lr_cutoff`; optionally k-space C₆ | `dispersion_energy`, plus `dispersion_energy_kspace` |

The [Observables](advanced/observables.md) page shows how to get each term and
per-atom quantity, and [Long-range electrostatics](advanced/electrostatics.md)
covers the k-space options.

## Why the long-range terms are explicit

Each message-passing layer reaches only to neighbours within the cutoff, so the
network sees the geometry only a few cutoffs around each atom (the total charge
and spin enter for the whole structure). Electrostatics and dispersion are
explicit terms with the correct 1/r and 1/r⁶ form instead of being learned from
data, so they also act between separated molecules. They reach as far as
`lr_cutoff`: 12 Å by default, so use 100 for isolated molecules, or the whole
periodic box with [k-space](advanced/electrostatics.md).

## The legacy `so3lr` model

The [original model](models.md#legacy-so3lr-model) computes the same four terms.
Its repulsion is a ZBL-type function fitted during training, and a single
Hirshfeld volume ratio per atom sets its dispersion. It covers eight elements
and was trained with real-space electrostatics only. See
[names across model generations](advanced/observables.md#names-differ-between-model-generations).

The [SO3LR Tauri preprint](https://doi.org/10.48550/arXiv.2611.06237) gives
the equations for every term in its "Model architecture" section, with details
in SI Sec. S1.
