# Repository guidance

This release contains SPS-GAN trajectory and video code. Keep implementation in
`src/sps_gan/` and regression tests in `src/tests/`. Keep exactly one example INI
in `config/`; `dataset/` contains only its README in version control.

Do not commit datasets, checkpoints, generated figures, outputs, credentials,
vendored simulators, or baseline implementations. Link external data sources in
`dataset/README.md`. Runtime artifacts belong in an ignored output directory.

Use explicit configuration objects and paths relative to the INI file. Package
imports must not read configuration, initialize training, or load datasets.
Preserve the model equations and loss scaling unless a change is intentional,
documented, and covered by a meaningful regression test. Describe only the
capabilities actually implemented here. Use SPS-GAN consistently in documentation.

Use Python 3.11 (`python.version`). Install with `uv sync --python 3.11 --locked`.
After code changes run `uv run --python 3.11 python -m unittest discover -s src/tests -v`.
Check CLI help with `uv run --python 3.11 sps-gan --help`. Keep `uv.lock` synchronized
with `pyproject.toml`. Do not replace this project's license or author attribution.
