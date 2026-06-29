"""Lumix layer selection and matrix extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax
import numpy as np
from flax import serialization


@dataclass(frozen=True)
class LayerSelector:
    """Select a Lumix layer by stable Flax name or qualified parameter path."""

    name: str | None = None
    path: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (self.name is None) == (self.path is None):
            raise ValueError("Specify exactly one of name or path.")

    @property
    def path_tuple(self) -> tuple[str, ...] | None:
        if self.path is None:
            return None
        if isinstance(self.path, str):
            return tuple(part for part in self.path.split("/") if part)
        return tuple(self.path)


def load_checkpoint_params(*, checkpoint_path: str | Path, model: Any, sample_x: Any):
    variables = model.init(jax.random.key(0), sample_x)
    params_template = variables["params"]
    return serialization.from_bytes(params_template, Path(checkpoint_path).read_bytes())


def extract_layer_matrix(*, model: Any, params: Any, sample_x: Any, selector: LayerSelector) -> np.ndarray:
    _, mutated = model.apply(
        {"params": params},
        sample_x,
        mutable=["lumix_inverse_design"],
    )
    collection = mutated.get("lumix_inverse_design", {})
    candidates = list(_iter_matrix_candidates(collection))
    if not candidates:
        raise ValueError(
            "No convertible Lumix layer matrices were exposed. "
            "Use UnitaryLinear or SubUnitaryLinear layers from this Lumix version."
        )
    selected_path = _resolve_selector(selector, [path for path, _ in candidates])
    for path, matrix in candidates:
        if path == selected_path:
            return np.asarray(jax.device_get(matrix), dtype=np.complex128)
    raise AssertionError("Resolved layer path was not found.")


def _iter_matrix_candidates(tree: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    for key, value in tree.items():
        path = (*prefix, str(key))
        if str(key) == "matrix":
            matrix = value[-1] if isinstance(value, tuple) else value
            yield prefix, matrix
        elif isinstance(value, Mapping):
            yield from _iter_matrix_candidates(value, path)


def _resolve_selector(selector: LayerSelector, paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if selector.path_tuple is not None:
        requested = selector.path_tuple
        if requested in paths:
            return requested
        available = ", ".join("/".join(path) for path in paths)
        raise ValueError(f"Layer path {'/'.join(requested)!r} not found. Available paths: {available}")

    assert selector.name is not None
    matches = [path for path in paths if path and path[-1] == selector.name]
    if not matches:
        available = ", ".join("/".join(path) for path in paths)
        raise ValueError(f"Layer name {selector.name!r} not found. Available paths: {available}")
    if len(matches) > 1:
        available = ", ".join("/".join(path) for path in matches)
        raise ValueError(f"Ambiguous layer selection {selector.name!r}. Matching paths: {available}")
    return matches[0]
