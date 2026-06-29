# Repository Guidelines

## Project Structure & Module Organization

`lumix` is a Python package for JAX/Flax optical neural network layers. Source code lives in `src/lumix/`, with public layer modules under `src/lumix/linen/`, pure numerical kernels under `src/lumix/functional/`, training routines under `src/lumix/training/`, and inverse-design/Tidy3D helpers under `src/lumix/inverse_design/`. Tests live in `tests/`; reference parity checks are in `tests/reference/`. Architecture decisions are recorded in `docs/adr/`. Large data assets, such as `simulation_data.hdf5`, should stay out of normal test paths unless a test explicitly needs them.

## Build, Test, and Development Commands

Use `uv` for Python tooling.

- `uv sync --extra dev`: install runtime dependencies plus pytest.
- `uv sync --extra dev --extra tidy3d`: include optional Tidy3D/GDS dependencies.
- `uv run pytest -q`: run the full test suite.
- `uv run pytest tests/test_inverse_design.py -q`: run a focused test module.
- `uv build`: build source and wheel distributions with Hatchling.

Agents in this local environment should prefix shell commands with `rtk`, for example `rtk uv run pytest -q`.

## Coding Style & Naming Conventions

Target Python 3.11+. Use 4-space indentation, explicit imports, and small functions that separate JAX array math from Flax `linen.Module` wrappers. Prefer immutable specs with `@dataclass(frozen=True)` or `flax.struct.dataclass` when values are part of a pytree/static configuration. Name tests and functions in `snake_case`; name modules by domain capability, such as `clements.py`, `routing.py`, or `tidy3d_builder.py`.

## Testing Guidelines

The suite uses `pytest` with JAX/Flax assertions. Add tests next to related coverage in `tests/test_*.py`; use descriptive `test_<behavior>` names. Keep fast unit tests independent of optional cloud services. For optional Tidy3D paths, prefer tests that validate generated specs or simulations locally and skip only when the dependency is unavailable.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add reference-exact training algorithms` and `Use field-interference adjoint for in-situ gradients`. Keep the first line concise and describe the behavioral change. Pull requests should include a short summary, the tests run, linked issues or ADRs when relevant, and screenshots/plots only for visual simulation output.

## Security & Configuration Tips

Do not commit API keys, cloud credentials, or generated solver outputs. Keep optional dependency usage guarded so the base package works after `uv sync --extra dev`.
