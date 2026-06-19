"""Metrics and metric-collection helpers for Lumix training loops."""

from collections.abc import Mapping, Sequence

import jax.numpy as jnp


def accuracy(probabilities_target: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    """Return classification accuracy for one-hot targets and predictions."""

    return jnp.mean(jnp.argmax(probabilities_target, axis=-1) == jnp.argmax(predictions, axis=-1))


def mean_squared_error(targets: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    """Return mean squared magnitude error for real or complex predictions."""

    error = predictions - targets
    return jnp.mean(jnp.square(jnp.abs(error)))


class Average:
    """NNX-like stateless average over metric values from one Linen apply call."""

    def __init__(self, values: Sequence[jnp.ndarray]):
        self.values = tuple(jnp.asarray(value) for value in values)

    def compute(self) -> jnp.ndarray:
        """Return the mean over all stored metric leaves."""

        if not self.values:
            return jnp.asarray(jnp.nan, dtype=jnp.float32)
        totals = [jnp.sum(value) for value in self.values]
        counts = [value.size for value in self.values]
        return jnp.sum(jnp.asarray(totals, dtype=jnp.float32)) / jnp.sum(jnp.asarray(counts, dtype=jnp.float32))


class AboveTarget:
    """Positive amount by which a value exceeds a target."""

    def __init__(self, target: float):
        self.target = target

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        """Return `max(value - target, 0)`."""

        array = jnp.asarray(value)
        return jnp.maximum(array - jnp.asarray(self.target, dtype=array.dtype), 0.0)


class MetricCollection:
    """Stateless reader for metrics emitted by Flax Linen collections.

    Use `MetricCollection.from_linen(updates["metrics"])` after a Linen
    `apply(..., mutable=["metrics"])` call. The collection does not own state;
    it only groups metric leaves from that single apply result.
    """

    def __init__(self, values_by_name: Mapping[str, Sequence[jnp.ndarray]]):
        self._values_by_name = {name: tuple(values) for name, values in values_by_name.items()}

    @classmethod
    def from_linen(cls, metrics: Mapping) -> "MetricCollection":
        """Create a metric collection from a nested Linen metrics tree."""

        values_by_name: dict[str, list[jnp.ndarray]] = {}

        def visit(node):
            if isinstance(node, Mapping):
                for name, value in node.items():
                    if isinstance(value, Mapping):
                        visit(value)
                    else:
                        values_by_name.setdefault(name, []).extend(_metric_leaves(value))

        visit(metrics)
        return cls(values_by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._values_by_name

    def __getitem__(self, name: str) -> Average:
        return Average(self._values_by_name[name])

    def mean(self, name: str, default: float | None = None) -> jnp.ndarray:
        """Return the average value for a named metric."""

        if name not in self._values_by_name:
            if default is None:
                raise KeyError(name)
            return jnp.asarray(default, dtype=jnp.float32)
        return self[name].compute()

    def above_target(self, name: str, target: float, default: float | None = None) -> jnp.ndarray:
        """Return positive amount by which a named metric exceeds a target."""

        return AboveTarget(target)(self.mean(name, default=default))


def _metric_leaves(value) -> list[jnp.ndarray]:
    if isinstance(value, tuple | list):
        leaves = []
        for item in value:
            leaves.extend(_metric_leaves(item))
        return leaves
    return [jnp.asarray(value)]
