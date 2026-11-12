# Models

The models are bundled with the package and selected by name.

## The current family

The SO3LR Tauri models, described in the
[SO3LR Tauri preprint](https://doi.org/10.48550/arXiv.2611.06237), come in three sizes:

- **`so3lr-tauri-m`** *(default)*: start here; the model we use for biomolecular MD.
- **`so3lr-tauri-l`**: the most accurate, recommended for reactions, and the
  slowest.
- **`so3lr-tauri-s`**: the fastest.

The Tauri models cover H to At (Z = 1–85), and the original `so3lr` covers H,
C, N, O, F, P, S and Cl. Structures with any other element are rejected with
an error.

| Model | Parameters | µs/atom/step, 1 A100 | µs/atom/step, 1 MI355X | ns/day, 50k atoms, 1 MI355X |
|---|---|---|---|---|
| `so3lr-tauri-s` | 411k | 1.7 | 0.69 | 2.5 |
| `so3lr-tauri-m` | 532k | 2.5 | 0.98 | 1.8 |
| `so3lr-tauri-l` | 1.79M | 8.7 | 3.2 | 0.54 |

MD cost on the flash engine for periodic liquid water (NVT, PME), from the
[SO3LR Tauri preprint](https://doi.org/10.48550/arXiv.2611.06237) (Computational
Performance and Table S25; ns/day at a 1 fs timestep). The legacy `so3lr`
(530k parameters) costs the same as `-m`. Test-set errors are given per subset
in Fig. 2 and Tables S5–S7 of the preprint, as there is no single overall
number.

The SO3LR Tauri weights are released under CC BY-NC 4.0 (non-commercial use);
the code and the legacy `so3lr` weights are MIT. See
[Licence](contributing.md#licence).

## Legacy SO3LR model

The model from the [JACS paper](https://doi.org/10.1021/jacs.5c09558) is kept,
as `so3lr`, to reproduce published results:

```python
from so3lr import So3lrCalculator

So3lrCalculator(model="so3lr")
```

## Names and versions

Names follow `so3lr-tauri-<size>[-<release>]`, where `<size>` is `s`, `m` or
`l` and `<release>` is the release date as `YYMM`:

| Name | Means | Stability |
|---|---|---|
| `so3lr-tauri-m` | the current M model | moves when a new release lands |
| `so3lr-tauri-m-2606` | one release | never changes |

The currently bundled releases are:

| Release | Canonical names |
|---|---|
| 2606 *(current, default)* | `so3lr-tauri-{s,m,l}-2606` |
| 2602 | `so3lr-tauri-{s,m,l}-2602` |
| Original | `so3lr` |

The 2602 models are the models after stage 4 of training, and 2606 the final
models after stage 5 (see the [preprint](https://doi.org/10.48550/arXiv.2611.06237)).

To pin a release for papers and benchmarks, use its dated name, e.g.
`so3lr-tauri-m-2606`, because the short names give different numbers after an
upgrade.

## Selecting a model

Use the same name in Python, on the command line and in settings files:

=== "Python / ASE"

    ```python
    from so3lr import So3lrCalculator

    calc = So3lrCalculator(model="so3lr-tauri-m")
    ```

=== "Command line"

    ```shell
    # opt / nvt / nvt-langevin / npt / nve / eval / tune-ewald
    so3lr nvt --input geometry.xyz --model so3lr-tauri-m
    ```

=== "Settings YAML"

    ```yaml
    model: so3lr-tauri-m   # alias of `model_path:`
    ```


## Custom and fine-tuned models

To use a fine-tuned or custom model, pass a path instead of a name:

```python
So3lrCalculator(model="/path/to/finetuned")
```

```shell
so3lr nvt --input geometry.xyz --model /path/to/finetuned
```

The directory needs a `hyperparameters.json` and either a `params.pkl` or an
orbax `checkpoints/` folder, which is what [fine-tuning](training/finetuning.md)
writes to its `--workdir`.

## Verifying the bundled files

The file `so3lr/models/SHA256SUMS` lists the SHA-256 of every bundled file. To
check them from the repository checkout:

```shell
cd so3lr/models && sha256sum -c SHA256SUMS   # macOS: shasum -a 256 -c SHA256SUMS
```

A match shows that the files are identical to the release.
