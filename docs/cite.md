# How to cite

If you use SO3LR, please cite the SO3LR Tauri preprint, the SO3LR paper and the
SO3krates paper, and give the software version and the dated model you used.

## Papers

```bibtex
@article{kabylda2026so3lr,
  title = {{SO3LR Tauri: Generally Applicable Force Fields for Reactive (Bio)Molecular Simulations Covering the Periodic Table}},
  author = {Kabylda, Adil and Br{\"{u}}nig, Florian N. and Gallegos, Miguel and Frank, J. Thorben and Moerman, Evgeny and Unke, Oliver T. and Chmiela, Stefan and M{\"{u}}ller, Klaus-Robert and Tkatchenko, Alexandre},
  journal = {arXiv preprint arXiv:2611.06237},
  year = {2026},
  doi = {10.48550/arXiv.2611.06237},
  url = {https://arxiv.org/abs/2611.06237},
}

@article{kabylda2025molecular,
  title={Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
  author={Kabylda, Adil and Frank, J Thorben and Su{\'a}rez-Dou, Sergio and Khabibrakhmanov, Almaz and Medrano Sandonas, Leonardo and Unke, Oliver T and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Tkatchenko, Alexandre},
  journal={Journal of the American Chemical Society},
  volume={147},
  number={37},
  pages={33723},
  year={2025},
  doi={10.1021/jacs.5c09558}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024},
  doi={10.1038/s41467-024-50620-6}
}
```

## Software version and model

Give the software version and the dated model together, for example:

```text
SO3LR v0.2.0 (so3lr-tauri-m-2606)
```

You can take the version from your installation (`so3lr.__version__`). For
reproducibility, also give the release tag or commit you used.

Write the dated name, e.g. `so3lr-tauri-m-2606`, not a short name such as
`so3lr-tauri-m`, which changes when a new release lands (see
[Models](models.md#names-and-versions)). If your results come from the original
`so3lr` model, please say so.

## Datasets

If you use the training or test data, please also cite the Zenodo archive:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22920714.svg)](https://doi.org/10.5281/zenodo.22920714)

For the data behind the legacy `so3lr` model, cite <https://doi.org/10.5281/zenodo.14779793>.

## Related software

SO3LR builds on the software below. Please cite it when it is central to your
results:

- [JAX-MD](https://github.com/jax-md/jax-md): the MD integrators and the neighbour-list engine
- [PySAGES](https://github.com/SSAGESLabs/PySAGES): enhanced sampling
- [torch-pme](https://github.com/lab-cosmo/torch-pme/) /
  [jax-pme](https://github.com/lab-cosmo/jax-pme/): k-space electrostatics
