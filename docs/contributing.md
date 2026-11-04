# Contributing

The code and models are still evolving, and your feedback is invaluable. We
would appreciate if you
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues)
for any errors, inconsistencies or feature ideas.

## Repository layout

| Path | Contents |
|---|---|
| `so3lr/` | Model, calculators and CLI |
| `so3lr/models/` | Bundled pretrained checkpoints, plus `SHA256SUMS` |
| `so3lr/model_registry.py` | Name → checkpoint resolution |
| `so3lr/cli/` | Command-line entry points |
| `so3lr/flash/` | The flash MD engine |
| `so3lr/mlff/` | The underlying SO3krates implementation |
| `so3lr/config/` | Default training and fine-tuning configs |
| `examples/` | Notebooks and their small input files |
| `tests/` | Unit tests |
| `tests/integration/` | End-to-end integration tests |
| `pyproject.toml`, `uv.lock` | Dependencies and the locked environment |

## Development setup

```shell
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
uv sync --locked --group dev
uv run python -c "import so3lr, jax"
```

Add the [extras](installation.md#optional-extras) (`--extra cuda12`,
`--extra tfds`, ...) when the change touches that stack.

### The lockfile

`uv.lock` defines the environment. Always pass `--locked`, and never edit
`uv.lock` by hand. To change a dependency:

1. Change `pyproject.toml`.
2. Run `uv lock --upgrade-package PACKAGE` (or `uv lock --upgrade`).
3. Commit the lockfile diff together with the change that needed it.

Git dependencies are pinned to full 40-character commits.

## Running tests

```shell
uv run pytest tests/                 # short suite (default)
uv run pytest tests/ --full          # everything, including slow tests
uv run pytest tests/ -n 16           # parallel (pytest-xdist); needs a large machine
```

The short suite skips tests marked `@pytest.mark.slow`. To run a single file
or test case:

```shell
uv run pytest tests/test_potential.py -v
uv run pytest tests/test_integration.py -v -k "nve_ensemble"
```

Integration tests can be sharded, as CI does:

```shell
uv run pytest tests/test_integration.py --shard 1/3
```

The flash engine's GPU tests are skipped without a suitable GPU.

## CI

GitHub Actions runs the short suite on every push to `main` and on every pull
request, in parallel groups in the locked environment. A push whose head
commit message contains `[full tests]` runs the full suite, and you can also
trigger either from the Actions tab. A packaging job builds and installs the
wheel and checks the bundled models against `SHA256SUMS`.

## Submitting changes

1. Branch off `main`.
2. Make your changes with clear commit messages.
3. Add or update tests.
4. Run the suite before opening the PR.
5. Open a pull request describing what changed and why.

## Adding tests

See
[`tests/integration/README.md`](https://github.com/general-molecular-simulations/so3lr/blob/main/tests/integration/README.md)
for how to add integration tests. Please also write unit tests for deeper
functionality.

## Adding a settings key

A new key must be listed in `YAML_DEFAULTS` or `KNOWN_YAML_KEYS_EXTRA` in
`so3lr/cli/so3lr_cli.py`, otherwise it is reported as a typo. It must also be
classified in `so3lr/flash/check.py` as `HANDLED`, `IRRELEVANT` or
`UNSUPPORTED`, otherwise MD falls back to the neighbour-list engine whenever it
is set (and `tests/test_flash.py` fails).

## Adding a model release

Model names are resolved in `so3lr/model_registry.py`. To add a release:

1. Put the checkpoint directory under `so3lr/models/<canonical-name>/`, holding
   `hyperparameters.json` and a `params.pkl` and/or an orbax `checkpoints/`
   folder.
2. Register it in `_CANONICAL`, using the `so3lr-tauri-<size>-<YYMM>` convention:

    ```python title="so3lr/model_registry.py"
    _CANONICAL = {
        # ... existing entries ...
        "so3lr-tauri-m-2610": ModelSpec(_MODELS / "so3lr-tauri-m-2610", family="v2"),
    }
    ```

3. Add the new files to `so3lr/models/SHA256SUMS`, with paths relative to
   `so3lr/models/`:

    ```shell
    cd so3lr/models
    sha256sum so3lr-tauri-m-2610/* >> SHA256SUMS   # macOS: shasum -a 256 so3lr-tauri-m-2610/* >> SHA256SUMS
    ```

4. If it becomes the recommended release, repoint `_FLOATING_ALIASES` and update
   `DEFAULT_MODEL`, which must stay a canonical name.
5. Update the [Models](https://so3lr.github.io/models/) page and the
   [changelog](https://so3lr.github.io/reference/changelog/).

The flash engine runs a model only if its hyperparameters match what the
kernels support (`so3lr/flash/check.py`); otherwise it uses the neighbour-list
engine.

!!! warning "Repointing an alias changes published numbers"
    Anyone whose script says `so3lr-tauri-m` gets the new checkpoint on their
    next upgrade. Please describe what changed in the changelog.

Extend `tests/test_model_registry.py` when adding a family or changing the
default.

## Documentation

- User-facing changes: update this site.
- Developer-facing changes: update `DEVELOPERS.md`.
- Use docstrings for all modules, classes and functions.

This site is a [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
project:

```shell
pip install -r requirements.txt
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static output in site/
```

When you add a CLI flag, a settings key or an observable, update the matching
reference table: [Command line](usage/cli.md),
[Settings reference](usage/settings.md) or
[Observables](advanced/observables.md).

## Licence

The code and the legacy SO3LR model weights are released under the MIT License,
and the SO3LR Tauri weights under CC BY-NC 4.0. See
[NOTICES.md](https://github.com/general-molecular-simulations/so3lr/blob/main/NOTICES.md)
for details.

## Contact

If you would like to see new features, add your own functionality, or share
ideas, please
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues)
or contact the maintainers.
