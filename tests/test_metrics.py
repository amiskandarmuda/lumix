import jax.numpy as jnp

from lumix.metrics import AboveTarget, Average, MetricCollection, mean_squared_error


def test_mean_squared_error_real_values():
    targets = jnp.array([1.0, 2.0, 4.0])
    predictions = jnp.array([1.0, 4.0, 1.0])
    error = mean_squared_error(targets, predictions)
    assert jnp.allclose(error, jnp.array(13.0 / 3.0))


def test_mean_squared_error_complex_values_uses_squared_magnitude():
    targets = jnp.array([1.0 + 1.0j, 2.0 + 0.0j])
    predictions = jnp.array([2.0 + 3.0j, 0.0 + 0.0j])
    error = mean_squared_error(targets, predictions)
    assert jnp.allclose(error, jnp.array(4.5))


def test_average_computes_mean_over_linen_metric_leaves():
    metric = Average(
        [
            jnp.array([0.1, 0.3], dtype=jnp.float32),
            jnp.array([[0.2], [0.4]], dtype=jnp.float32),
        ]
    )

    assert jnp.allclose(metric.compute(), jnp.array(0.25, dtype=jnp.float32))


def test_average_accepts_scalar_values():
    metric = Average([0.1, jnp.array(0.3)])

    assert jnp.allclose(metric.compute(), jnp.array(0.2, dtype=jnp.float32))


def test_metric_collection_reads_nested_linen_metric_collection():
    linen_metrics = {
        "unitary_0": {"routing_leakage": (jnp.array(0.1),)},
        "unitary_1": {"routing_leakage": (jnp.array([0.2, 0.4]),)},
    }

    metrics = MetricCollection.from_linen(linen_metrics)

    assert "routing_leakage" in metrics
    assert jnp.allclose(metrics["routing_leakage"].compute(), jnp.array((0.1 + 0.2 + 0.4) / 3.0))


def test_metric_collection_returns_default_for_missing_metric():
    metrics = MetricCollection.from_linen({})

    assert jnp.allclose(metrics.mean("routing_leakage", default=0.5), jnp.array(0.5))


def test_above_target_returns_positive_amount_above_target():
    above_target = AboveTarget(0.1)

    assert jnp.allclose(above_target(jnp.array(0.25)), jnp.array(0.15))
    assert jnp.allclose(above_target(jnp.array(0.05)), jnp.array(0.0))
    assert jnp.allclose(above_target(0.25), jnp.array(0.15))


def test_metric_collection_above_target_uses_metric_mean():
    linen_metrics = {
        "unitary_0": {"routing_leakage": (jnp.array(0.12),)},
        "unitary_1": {"routing_leakage": (jnp.array(0.18),)},
    }
    metrics = MetricCollection.from_linen(linen_metrics)

    assert jnp.allclose(metrics.above_target("routing_leakage", 0.1), jnp.array(0.05, dtype=jnp.float32))
