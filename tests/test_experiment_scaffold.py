import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.common.mnist_pca import fit_pca16_dataset
from experiments.common.mnist_pca import is_current_pca_dataset
from experiments.common.models import (
    SubUnitarySurrogateConfig,
    build_subunitary_surrogate,
    extract_inverse_design_matrices,
    subunitary_surrogate_config_from_mapping,
)
from experiments.common.robustness import noisy_surrogate_forward
from experiments.common.robustness import relative_frobenius_perturbation
from experiments.common.training import routing_regularized_loss
from experiments.common.training import class_margin_loss
from experiments.common.training import distillation_kl_logits
from experiments.common.training import make_matrix_noise_distilled_train_step
from experiments.common.training import make_matrix_noise_regularized_train_step
from experiments.common.training import make_routing_regularized_train_step
from experiments.common.training import create_shared_prefix_state
from experiments.common.training import fit_shared_prefix_distilled_logits
from experiments.common.training import fit_shared_prefix_routing_regularized_logits
from experiments.common.training import make_shared_prefix_distilled_train_step
from experiments.common.training import make_shared_prefix_routing_regularized_train_step
from experiments.scripts.train_shared_prefix_surrogate import apply_cli_overrides
from experiments.scripts.train_shared_prefix_surrogate import parse_prefix_weights
from experiments.scripts.train_matrix_noise_robust_surrogate import checkpoint_selection_value
from experiments.scripts.train_matrix_noise_robust_surrogate import load_params_into_state
from experiments.scripts.calibrate_phase_bias_under_matrix_error import calibrated_noisy_forward
from experiments.scripts.calibrate_phase_bias_under_matrix_error import phase_bias_multiplier
from experiments.scripts.train_progressive_shared_prefix_surrogate import adaptive_prefix_weights
from experiments.scripts.train_progressive_shared_prefix_surrogate import parse_float_list
from experiments.scripts.train_progressive_shared_prefix_surrogate import progressive_growth_depths
from lumix.state import create_state
from lumix.functional.subunitary import insertion_loss_bounds
from lumix.functional.routing import routing_mask


def test_fit_pca16_dataset_projects_images_to_configured_width():
    train_images = np.arange(20 * 4 * 4, dtype=np.float32).reshape(20, 4, 4)
    test_images = np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)

    dataset = fit_pca16_dataset(train_images, np.arange(20) % 10, test_images, np.arange(8) % 10, components=3)

    assert dataset.x_train.shape == (20, 3)
    assert dataset.x_test.shape == (8, 3)
    assert dataset.y_train.shape == (20, 10)
    assert dataset.y_test.shape == (8, 10)
    assert dataset.mean.shape == (16,)
    assert dataset.components.shape == (3, 16)
    assert dataset.x_train.dtype == np.float32
    assert np.all(np.isfinite(dataset.x_train))


def test_fit_pca16_dataset_standardizes_then_minmax_maps_train_features():
    rng = np.random.default_rng(7)
    train_images = rng.integers(0, 256, size=(64, 4, 4), dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(16, 4, 4), dtype=np.uint8)

    dataset = fit_pca16_dataset(train_images, np.arange(64) % 10, test_images, np.arange(16) % 10, components=4)

    assert np.allclose(dataset.x_train.min(axis=0), 0.0, atol=1e-6)
    assert np.allclose(dataset.x_train.max(axis=0), 1.0, atol=1e-6)
    assert dataset.feature_min.shape == (4,)
    assert dataset.feature_range.shape == (4,)


def test_fit_pca16_dataset_can_emit_zscore_features():
    rng = np.random.default_rng(11)
    train_images = rng.integers(0, 256, size=(64, 4, 4), dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(16, 4, 4), dtype=np.uint8)

    dataset = fit_pca16_dataset(
        train_images,
        np.arange(64) % 10,
        test_images,
        np.arange(16) % 10,
        components=4,
        preprocessing="pca_zscore",
    )

    assert dataset.preprocessing == "pca_zscore"
    assert np.allclose(dataset.x_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(dataset.x_train.std(axis=0), 1.0, atol=1e-6)
    assert np.all(np.isfinite(dataset.x_test))


def test_pca_dataset_cache_can_validate_expected_preprocessing(tmp_path):
    cache_path = tmp_path / "encoded_pca.npz"
    np.savez_compressed(cache_path, preprocessing=np.asarray("pca_zscore_phase"))

    assert is_current_pca_dataset(cache_path, expected_preprocessing="pca_zscore_phase")
    assert not is_current_pca_dataset(cache_path)


def test_default_mnist_case_uses_zscore_phase_open_loss():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json"
    )
    config = json.loads(config_path.read_text())

    assert config["dataset"]["preprocessing"] == "pca_zscore"
    assert config["dataset"]["processed_path"] == "../../../datasets/mnist_pca16/processed/mnist_pca16_zscore.npz"
    assert config["model"]["loss_db"] == [0.0, None]
    assert config["model"]["phase_scale"] == 0.5
    assert config["model"]["nonlinearity"] == "repeated_phase_mask"
    assert config["model"]["readout_port_start"] is None
    assert config["training"]["loss_guard_db"] is None
    assert config["training"]["loss_guard_weight"] == 0.0


def test_shared_prefix_runner_parses_explicit_prefix_weights():
    weights = parse_prefix_weights("0.1,0.2,1.0", (1, 2, 3))

    assert weights == (0.1, 0.2, 1.0)


def test_shared_prefix_runner_rejects_wrong_weight_count():
    with np.testing.assert_raises(ValueError):
        parse_prefix_weights("0.1,1.0", (1, 2, 3))


def test_shared_prefix_runner_applies_parameter_overrides():
    config = {
        "model": {
            "layers": 5,
            "width": 16,
            "loss_db": [0.0, None],
            "routing_limit": 4,
            "classes": 10,
        },
        "training": {
            "learning_rate": 0.005,
            "routing_penalty_weight": 0.5,
            "routing_leakage_target": 0.1,
        },
    }

    apply_cli_overrides(
        config,
        routing_limit=7,
        routing_weight=0.25,
        routing_target=0.2,
        learning_rate=0.003,
    )

    assert config["model"]["routing_limit"] == 7
    assert config["training"]["routing_penalty_weight"] == 0.25
    assert config["training"]["routing_leakage_target"] == 0.2
    assert config["training"]["learning_rate"] == 0.003


def test_progressive_runner_parses_float_lists_with_expected_count():
    values = parse_float_list("0.8,0.9,0.943", expected_count=3, label="target accuracies")

    assert values == (0.8, 0.9, 0.943)


def test_progressive_runner_weights_underperforming_prefixes_more_heavily():
    weights = adaptive_prefix_weights(
        prefix_depths=(1, 2, 3),
        target_accuracies=(0.8, 0.9, 0.94),
        previous_metrics={
            "prefix_1_val_accuracy": 0.82,
            "prefix_2_val_accuracy": 0.86,
            "prefix_3_val_accuracy": 0.70,
        },
        base_weight=0.1,
        deficit_scale=10.0,
        deficit_power=1.0,
        new_depth_boost=0.0,
    )

    assert weights[2] > weights[1] > weights[0]
    assert np.isclose(sum(weights), 1.0)


def test_progressive_runner_can_skip_growth_depths_for_checkpoint_continuation():
    assert progressive_growth_depths(max_depth=4, skip_growth_stages=False) == (1, 2, 3, 4)
    assert progressive_growth_depths(max_depth=4, skip_growth_stages=True) == ()


def test_subunitary_surrogate_outputs_logits_for_three_layers():
    config = SubUnitarySurrogateConfig(width=16, layers=3, loss_db=(2.0, 3.0), classes=10)
    model = build_subunitary_surrogate(config)
    values = jnp.ones((4, 16), dtype=jnp.float32)

    variables = model.init(jax.random.key(0), values)
    logits = model.apply(variables, values)

    assert logits.shape == (4, 10)


def test_subunitary_surrogate_uses_unit_amplitude_input_fields():
    config = SubUnitarySurrogateConfig(width=16, layers=3, loss_db=(2.0, 3.0), classes=10)
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 32, dtype=jnp.float32).reshape(2, 16)
    variables = model.init(jax.random.key(7), values)

    fields = model.apply(variables, values, method=type(model).input_fields)

    assert fields.shape == (2, 16)
    assert fields.dtype == jnp.complex64
    assert jnp.allclose(fields, jnp.ones_like(fields))


def test_subunitary_surrogate_builds_repeated_phase_mask_from_original_inputs():
    config = SubUnitarySurrogateConfig(width=16, layers=3, loss_db=(2.0, 3.0), classes=10)
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 32, dtype=jnp.float32).reshape(2, 16)
    variables = model.init(jax.random.key(8), values)

    phase_mask = model.apply(variables, values, method=type(model).repeated_phase_mask)

    assert phase_mask.shape == (2, 16)
    assert phase_mask.dtype == jnp.complex64
    assert jnp.allclose(jnp.abs(phase_mask), jnp.ones_like(values))
    assert jnp.allclose(phase_mask, jnp.exp(1j * jnp.pi * values).astype(jnp.complex64))


def test_subunitary_surrogate_appends_constant_phase_zero_bias_port():
    config = SubUnitarySurrogateConfig(
        width=17,
        layers=1,
        loss_db=(0.0, None),
        classes=10,
        input_amplitude=0.25,
        phase_scale=0.5,
        bias_ports=1,
        readout="temperature_softmax",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 32, dtype=jnp.float32).reshape(2, 16)
    variables = model.init(jax.random.key(40), values)

    input_fields = model.apply(variables, values, method=type(model).input_fields)
    phase_mask = model.apply(variables, values, method=type(model).repeated_phase_mask)
    logits = model.apply(variables, values)

    assert input_fields.shape == (2, 17)
    assert phase_mask.shape == (2, 17)
    assert logits.shape == (2, 10)
    assert jnp.allclose(input_fields, jnp.full((2, 17), 0.25, dtype=jnp.complex64))
    assert jnp.allclose(phase_mask[..., :16], jnp.exp(0.5j * values).astype(jnp.complex64))
    assert jnp.allclose(phase_mask[..., 16], jnp.ones((2,), dtype=jnp.complex64))


def test_subunitary_surrogate_can_encode_phase_once_without_data_repetition():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=3,
        loss_db=(2.0, 3.0),
        classes=10,
        nonlinearity="input_phase_once",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 32, dtype=jnp.float32).reshape(2, 16)
    variables = model.init(jax.random.key(28), values)

    first_layer_mask = model.apply(variables, values, 0, method=type(model).phase_mask_for_layer)
    second_layer_mask = model.apply(variables, values, 1, method=type(model).phase_mask_for_layer)

    assert jnp.allclose(first_layer_mask, jnp.exp(1j * jnp.pi * values).astype(jnp.complex64))
    assert jnp.allclose(second_layer_mask, jnp.ones_like(first_layer_mask))


def test_temperature_softmax_readout_scales_uniform_class_intensities():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=0,
        loss_db=(0.0, 1.5),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
        init_gamma=10.0,
    )
    model = build_subunitary_surrogate(config)
    values = jnp.zeros((2, 16), dtype=jnp.float32)
    variables = model.init(jax.random.key(9), values)

    logits, aux = model.apply(variables, values, return_aux=True)

    assert logits.shape == (2, 10)
    assert jnp.allclose(aux["intensities"], jnp.full((2, 16), 0.0625, dtype=jnp.float32))
    assert jnp.allclose(aux["gamma"], 10.0)
    assert jnp.allclose(logits, jnp.full((2, 10), 0.625, dtype=jnp.float32))


def test_temperature_softmax_readout_uses_center_ports_by_default():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=0,
        loss_db=(0.0, 1.5),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
        init_gamma=10.0,
    )
    model = build_subunitary_surrogate(config)
    values = jnp.zeros((1, 16), dtype=jnp.float32)
    params = model.init(jax.random.key(35), values)["params"]
    custom_fields = jnp.sqrt(jnp.arange(16, dtype=jnp.float32).reshape(1, 16) + 1.0).astype(jnp.complex64)

    logits, aux = model.apply(
        {"params": params},
        custom_fields,
        return_aux=True,
        method=type(model).readout_fields,
    )

    assert logits.shape == (1, 10)
    assert jnp.allclose(aux["intensities"], jnp.arange(16, dtype=jnp.float32).reshape(1, 16) + 1.0)
    assert jnp.allclose(logits, 10.0 * jnp.arange(4, 14, dtype=jnp.float32).reshape(1, 10))


def test_intensity_logits_readout_uses_center_ports_by_default():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=0,
        loss_db=(0.0, 1.5),
        classes=10,
        readout="intensity_logits",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.zeros((1, 16), dtype=jnp.float32)
    variables = model.init(jax.random.key(36), values)
    custom_fields = jnp.sqrt(jnp.arange(16, dtype=jnp.float32).reshape(1, 16) + 1.0).astype(jnp.complex64)

    logits, aux = model.apply(
        variables,
        custom_fields,
        return_aux=True,
        method=type(model).readout_fields,
    )

    assert logits.shape == (1, 10)
    assert jnp.allclose(aux["intensities"], jnp.arange(16, dtype=jnp.float32).reshape(1, 16) + 1.0)
    assert jnp.allclose(logits, jnp.arange(4, 14, dtype=jnp.float32).reshape(1, 10))


def test_config_accepts_explicit_readout_port_start():
    config = subunitary_surrogate_config_from_mapping(
        {
            "width": 16,
            "layers": 3,
            "loss_db": [2.0, None],
            "classes": 10,
            "readout_port_start": 3,
        }
    )

    assert config.readout_port_start == 3


def test_subunitary_surrogate_emits_routing_metrics_when_limited():
    config = SubUnitarySurrogateConfig(width=16, layers=3, loss_db=(2.0, 3.0), classes=10, routing_limit=7)
    model = build_subunitary_surrogate(config)
    values = jnp.ones((4, 16), dtype=jnp.float32)

    variables = model.init(jax.random.key(2), values)
    _, metrics = model.apply(variables, values, mutable=["metrics"])

    assert sorted(metrics["metrics"]) == ["optical_0", "optical_1", "optical_2"]
    assert all("routing_leakage" in metrics["metrics"][name] for name in metrics["metrics"])


def test_routing_regularized_loss_penalizes_excess_leakage_without_hard_masking():
    config = SubUnitarySurrogateConfig(width=16, layers=3, loss_db=(2.0, 3.0), classes=10, routing_limit=7)
    model = build_subunitary_surrogate(config)
    values = jnp.ones((4, 16), dtype=jnp.float32)
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    params = model.init(jax.random.key(3), values)["params"]

    total, parts = routing_regularized_loss(
        model,
        params,
        values,
        labels,
        routing_weight=2.0,
        routing_target=0.0,
    )
    unweighted_total, unweighted_parts = routing_regularized_loss(
        model,
        params,
        values,
        labels,
        routing_weight=0.0,
        routing_target=0.0,
    )

    assert float(parts.routing_leakage) > 0.0
    assert jnp.allclose(parts.routing_excess, parts.routing_leakage)
    assert jnp.allclose(total, parts.cross_entropy + 2.0 * parts.routing_excess)
    assert jnp.allclose(unweighted_total, unweighted_parts.cross_entropy)


def test_routing_regularized_loss_includes_insertion_loss_guard_when_configured():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=0,
        loss_db=(0.0, 1.5),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.zeros((4, 16), dtype=jnp.float32)
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    params = model.init(jax.random.key(10), values)["params"]

    total, parts = routing_regularized_loss(
        model,
        params,
        values,
        labels,
        routing_weight=0.0,
        routing_target=0.0,
        loss_guard_db=-1.0,
        loss_guard_weight=5.0,
    )

    assert jnp.allclose(parts.mean_output_power, 1.0)
    assert jnp.allclose(parts.mean_insertion_loss_db, 0.0, atol=1e-6)
    assert jnp.allclose(parts.loss_excess, 1.0, atol=1e-6)
    assert jnp.allclose(total, parts.cross_entropy + 5.0 * parts.loss_excess**2)


def test_routing_regularized_train_step_updates_without_mutable_metric_leak():
    config = SubUnitarySurrogateConfig(width=16, layers=1, loss_db=(2.0, 3.0), classes=10, routing_limit=7)
    model = build_subunitary_surrogate(config)
    values = jnp.ones((4, 16), dtype=jnp.float32)
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    state = create_state(model, jax.random.key(4), values, learning_rate=1e-3)
    train_step = make_routing_regularized_train_step(model, routing_weight=2.0, routing_target=0.05)

    next_state, loss, score, parts = train_step(state, values, labels)

    assert next_state.step == state.step + 1
    assert jnp.isfinite(loss)
    assert jnp.isfinite(score)
    assert jnp.isfinite(parts.cross_entropy)
    assert jnp.isfinite(parts.routing_leakage)
    assert jnp.isfinite(parts.routing_excess)
    assert jnp.isfinite(parts.mean_insertion_loss_db)
    assert jnp.isfinite(parts.mean_output_power)


def test_relative_frobenius_perturbation_matches_requested_error_norm():
    matrix = jnp.eye(4, dtype=jnp.complex64)

    perturbed = relative_frobenius_perturbation(matrix, jax.random.key(41), relative_error=0.2)
    clean = relative_frobenius_perturbation(matrix, jax.random.key(41), relative_error=0.0)

    assert jnp.allclose(clean, matrix)
    error_ratio = jnp.linalg.norm(perturbed - matrix) / jnp.linalg.norm(matrix)
    assert jnp.allclose(error_ratio, 0.2, atol=1e-6)


def test_noisy_surrogate_forward_matches_clean_forward_at_zero_error():
    config = SubUnitarySurrogateConfig(width=4, layers=2, loss_db=(0.0, None), classes=3)
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32).reshape(2, 4)
    params = model.init(jax.random.key(42), values)["params"]

    clean_logits = model.apply({"params": params}, values)
    noisy_logits = noisy_surrogate_forward(
        model,
        params,
        values,
        jax.random.key(43),
        relative_error=0.0,
    )

    assert jnp.allclose(noisy_logits, clean_logits, atol=1e-6)


def test_phase_bias_multiplier_uses_pi_scaled_fixed_offsets():
    bias = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)

    multiplier = phase_bias_multiplier(bias, phase_bias_scale=jnp.pi)

    assert multiplier.shape == (3,)
    assert jnp.allclose(multiplier, jnp.exp(1j * jnp.pi * bias).astype(jnp.complex64))


def test_calibrated_noisy_forward_matches_clean_forward_with_zero_bias_and_error():
    config = SubUnitarySurrogateConfig(width=4, layers=2, loss_db=(0.0, None), classes=3)
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32).reshape(2, 4)
    params = model.init(jax.random.key(50), values)["params"]
    phase_bias = jnp.zeros((2, 4), dtype=jnp.float32)

    clean_logits = model.apply({"params": params}, values)
    calibrated_logits = calibrated_noisy_forward(
        model,
        params,
        values,
        phase_bias,
        jax.random.key(51),
        relative_error=0.0,
        phase_bias_scale=jnp.pi,
        per_layer=True,
    )

    assert jnp.allclose(calibrated_logits, clean_logits, atol=1e-6)


def test_matrix_noise_regularized_train_step_updates_with_noisy_loss():
    config = SubUnitarySurrogateConfig(width=4, layers=1, loss_db=(0.0, None), classes=3, routing_limit=2)
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4)
    labels = jnp.eye(3, dtype=jnp.float32)[jnp.arange(4) % 3]
    state = create_state(model, jax.random.key(44), values, learning_rate=1e-3)
    train_step = make_matrix_noise_regularized_train_step(
        model,
        routing_weight=0.5,
        routing_target=0.1,
        relative_error=0.2,
        noise_samples=2,
        noisy_weight=1.0,
    )

    next_state, loss, score, parts, noisy_cross_entropy = train_step(
        state,
        values,
        labels,
        jax.random.key(45),
    )

    assert next_state.step == state.step + 1
    assert jnp.isfinite(loss)
    assert jnp.isfinite(score)
    assert jnp.isfinite(parts.cross_entropy)
    assert jnp.isfinite(noisy_cross_entropy)


def test_class_margin_loss_penalizes_logits_below_target_margin():
    labels = jnp.eye(3, dtype=jnp.float32)[jnp.asarray([0, 1])]
    strong_logits = jnp.asarray([[3.0, 1.0, 0.0], [0.0, 2.5, 0.5]], dtype=jnp.float32)
    weak_logits = jnp.asarray([[1.1, 1.0, 0.0], [0.0, 1.2, 0.5]], dtype=jnp.float32)

    assert jnp.allclose(class_margin_loss(labels, strong_logits, target_margin=1.0), 0.0)
    assert float(class_margin_loss(labels, weak_logits, target_margin=1.0)) > 0.0


def test_matrix_noise_distilled_train_step_updates_from_teacher_logits():
    config = SubUnitarySurrogateConfig(width=4, layers=1, loss_db=(0.0, None), classes=3, routing_limit=2)
    model = build_subunitary_surrogate(config)
    values = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4)
    labels = jnp.eye(3, dtype=jnp.float32)[jnp.arange(4) % 3]
    teacher_logits = labels * 3.0
    state = create_state(model, jax.random.key(46), values, learning_rate=1e-3)
    train_step = make_matrix_noise_distilled_train_step(
        model,
        routing_weight=0.5,
        routing_target=0.1,
        relative_error=0.2,
        noise_samples=2,
        noisy_weight=1.0,
        distillation_weight=0.7,
        distillation_temperature=2.0,
        margin_weight=0.3,
        margin_target=0.1,
    )

    next_state, loss, score, parts, noisy_cross_entropy, distillation_kl, margin_loss = train_step(
        state,
        values,
        labels,
        teacher_logits,
        jax.random.key(47),
    )

    assert next_state.step == state.step + 1
    assert jnp.isfinite(loss)
    assert jnp.isfinite(score)
    assert jnp.isfinite(parts.cross_entropy)
    assert jnp.isfinite(noisy_cross_entropy)
    assert jnp.isfinite(distillation_kl)
    assert jnp.isfinite(margin_loss)


def test_checkpoint_selection_value_applies_clean_accuracy_floor():
    passing = {"val_accuracy": 0.93, "val_noisy_accuracy": 0.91}
    failing = {"val_accuracy": 0.90, "val_noisy_accuracy": 0.95}

    assert checkpoint_selection_value(passing, "val_noisy_accuracy", clean_accuracy_floor=0.92) == 0.91
    assert checkpoint_selection_value(failing, "val_noisy_accuracy", clean_accuracy_floor=0.92) == float("-inf")


def test_load_params_into_state_restores_serialized_params(tmp_path):
    config = SubUnitarySurrogateConfig(width=4, layers=1, loss_db=(0.0, None), classes=3)
    model = build_subunitary_surrogate(config)
    values = jnp.ones((2, 4), dtype=jnp.float32)
    state = create_state(model, jax.random.key(48), values, learning_rate=1e-3)
    restored_state = create_state(model, jax.random.key(49), values, learning_rate=1e-3)
    params_path = tmp_path / "params.msgpack"

    from flax import serialization

    params_path.write_bytes(serialization.to_bytes(state.params))
    loaded_state = load_params_into_state(restored_state, params_path)

    original_flat = jax.tree.leaves(state.params)
    loaded_flat = jax.tree.leaves(loaded_state.params)
    assert all(jnp.allclose(original, loaded) for original, loaded in zip(original_flat, loaded_flat))


def test_shared_prefix_state_initializes_shared_layers_and_depth_readouts():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=4,
        loss_db=(0.0, None),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.ones((3, 16), dtype=jnp.float32)

    state = create_shared_prefix_state(
        model,
        jax.random.key(31),
        values,
        learning_rate=1e-3,
        prefix_depths=(1, 2, 3, 4),
    )

    param_names = set(state.params)
    assert {f"subunitary_{index}" for index in range(4)}.issubset(param_names)
    assert {f"readout_depth_{depth}" for depth in range(1, 5)}.issubset(param_names)

    logits_depth_1, aux_depth_1 = model.apply(
        {"params": state.params},
        values,
        return_aux=True,
        depth=1,
        readout_name="readout_depth_1",
    )
    logits_depth_4, aux_depth_4 = model.apply(
        {"params": state.params},
        values,
        return_aux=True,
        depth=4,
        readout_name="readout_depth_4",
    )

    assert logits_depth_1.shape == (3, 10)
    assert logits_depth_4.shape == (3, 10)
    assert jnp.isfinite(aux_depth_1["gamma"])
    assert jnp.isfinite(aux_depth_4["gamma"])


def test_shared_prefix_train_step_reports_one_metric_per_depth():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=3,
        loss_db=(0.0, None),
        classes=10,
        routing_limit=4,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.ones((5, 16), dtype=jnp.float32)
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(5) % 10]
    state = create_shared_prefix_state(
        model,
        jax.random.key(32),
        values,
        learning_rate=1e-3,
        prefix_depths=(1, 2, 3),
    )
    train_step = make_shared_prefix_routing_regularized_train_step(
        model,
        prefix_depths=(1, 2, 3),
        routing_weight=0.5,
        routing_target=0.1,
    )

    next_state, loss, scores, parts = train_step(state, values, labels)

    assert next_state.step == state.step + 1
    assert scores.shape == (3,)
    assert parts.cross_entropy.shape == (3,)
    assert parts.routing_leakage.shape == (3,)
    assert parts.mean_insertion_loss_db.shape == (3,)
    assert jnp.all(jnp.isfinite(scores))
    assert jnp.isfinite(loss)


def test_distillation_kl_logits_is_zero_for_matching_teacher_logits():
    teacher_logits = jnp.asarray(
        [
            [2.0, 1.0, -1.0],
            [-0.5, 0.25, 1.75],
        ],
        dtype=jnp.float32,
    )
    shifted_student_logits = jnp.asarray(
        [
            [-1.0, 1.0, 2.0],
            [1.75, 0.25, -0.5],
        ],
        dtype=jnp.float32,
    )

    matching = distillation_kl_logits(teacher_logits, teacher_logits, temperature=2.0)
    shifted = distillation_kl_logits(shifted_student_logits, teacher_logits, temperature=2.0)

    assert jnp.allclose(matching, 0.0, atol=1e-6)
    assert float(shifted) > 0.0


def test_shared_prefix_distilled_train_step_reports_one_kd_metric_per_depth():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=2,
        loss_db=(0.0, None),
        classes=10,
        routing_limit=4,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    values = jnp.ones((5, 16), dtype=jnp.float32)
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(5) % 10]
    teacher_logits = jnp.stack([labels * 2.0, labels * 3.0])
    state = create_shared_prefix_state(
        model,
        jax.random.key(37),
        values,
        learning_rate=1e-3,
        prefix_depths=(1, 2),
    )
    train_step = make_shared_prefix_distilled_train_step(
        model,
        prefix_depths=(1, 2),
        routing_weight=0.5,
        routing_target=0.1,
        distillation_alpha=0.7,
        distillation_temperature=2.0,
    )

    next_state, loss, scores, parts, distillation_kl = train_step(state, values, labels, teacher_logits)

    assert next_state.step == state.step + 1
    assert scores.shape == (2,)
    assert parts.cross_entropy.shape == (2,)
    assert distillation_kl.shape == (2,)
    assert jnp.all(jnp.isfinite(distillation_kl))
    assert jnp.isfinite(loss)


def test_shared_prefix_fit_history_records_depth_specific_validation_metrics():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=2,
        loss_db=(0.0, None),
        classes=10,
        routing_limit=4,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    train_x = jnp.ones((8, 16), dtype=jnp.float32)
    train_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(8) % 10]
    test_x = jnp.ones((4, 16), dtype=jnp.float32)
    test_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    state = create_shared_prefix_state(
        model,
        jax.random.key(33),
        train_x[:1],
        learning_rate=1e-3,
        prefix_depths=(1, 2),
    )

    final_state, history = fit_shared_prefix_routing_regularized_logits(
        model,
        state,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs=2,
        batch_size=4,
        prefix_depths=(1, 2),
        routing_weight=0.5,
        routing_target=0.1,
        select_best_checkpoint=True,
        checkpoint_epochs=(1, 2),
        seed=33,
    )

    assert final_state.step > state.step
    assert history["metadata"]["prefix_depths"] == [1, 2]
    assert len(history["epoch"]) == 2
    assert len(history["prefix_1_val_accuracy"]) == 2
    assert len(history["prefix_2_val_mean_insertion_loss_db"]) == 2
    assert "prefix_1_val_accuracy" in history["selected_metrics"]
    assert "prefix_2_val_accuracy" in history["selected_metrics"]


def test_shared_prefix_distilled_fit_records_depth_specific_distillation_metrics():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=2,
        loss_db=(0.0, None),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    train_x = jnp.ones((8, 16), dtype=jnp.float32)
    train_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(8) % 10]
    test_x = jnp.ones((4, 16), dtype=jnp.float32)
    test_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    teacher_train_logits = jnp.stack([train_y * 2.0, train_y * 3.0])
    teacher_test_logits = jnp.stack([test_y * 2.0, test_y * 3.0])
    state = create_shared_prefix_state(
        model,
        jax.random.key(38),
        train_x[:1],
        learning_rate=1e-3,
        prefix_depths=(1, 2),
    )

    _, history = fit_shared_prefix_distilled_logits(
        model,
        state,
        train_x,
        train_y,
        test_x,
        test_y,
        teacher_train_logits,
        teacher_test_logits,
        epochs=2,
        batch_size=4,
        prefix_depths=(1, 2),
        routing_weight=0.0,
        routing_target=0.0,
        distillation_alpha=0.6,
        distillation_temperature=2.0,
        select_best_checkpoint=True,
        checkpoint_epochs=(1, 2),
        selection_metric="prefix_2_val_accuracy",
        seed=38,
    )

    assert history["metadata"]["distillation_alpha"] == 0.6
    assert history["metadata"]["distillation_temperature"] == 2.0
    assert len(history["prefix_1_distillation_kl"]) == 2
    assert len(history["prefix_2_val_distillation_kl"]) == 2
    assert "prefix_1_val_distillation_kl" in history["selected_metrics"]
    assert "prefix_2_val_distillation_kl" in history["selected_metrics"]


def test_shared_prefix_fit_can_select_checkpoint_by_specific_prefix_metric():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=2,
        loss_db=(0.0, None),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    train_x = jnp.ones((8, 16), dtype=jnp.float32)
    train_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(8) % 10]
    test_x = jnp.ones((4, 16), dtype=jnp.float32)
    test_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    state = create_shared_prefix_state(
        model,
        jax.random.key(34),
        train_x[:1],
        learning_rate=1e-3,
        prefix_depths=(1, 2),
    )

    _, history = fit_shared_prefix_routing_regularized_logits(
        model,
        state,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs=3,
        batch_size=4,
        prefix_depths=(1, 2),
        routing_weight=0.0,
        routing_target=0.0,
        select_best_checkpoint=True,
        checkpoint_epochs=(1, 2, 3),
        selection_metric="prefix_2_val_accuracy",
        seed=34,
    )

    selected_index = history["epoch"].index(history["selected_epoch"])
    assert history["metadata"]["selection_metric"] == "prefix_2_val_accuracy"
    assert history["prefix_2_val_accuracy"][selected_index] == max(history["prefix_2_val_accuracy"])


def test_shared_prefix_fit_records_and_selects_min_target_margin():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=2,
        loss_db=(0.0, None),
        classes=10,
        input_amplitude=0.25,
        readout="temperature_softmax",
        layer_name_prefix="subunitary",
    )
    model = build_subunitary_surrogate(config)
    train_x = jnp.ones((8, 16), dtype=jnp.float32)
    train_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(8) % 10]
    test_x = jnp.ones((4, 16), dtype=jnp.float32)
    test_y = jnp.eye(10, dtype=jnp.float32)[jnp.arange(4) % 10]
    state = create_shared_prefix_state(
        model,
        jax.random.key(39),
        train_x[:1],
        learning_rate=1e-3,
        prefix_depths=(1, 2),
    )

    _, history = fit_shared_prefix_routing_regularized_logits(
        model,
        state,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs=2,
        batch_size=4,
        prefix_depths=(1, 2),
        routing_weight=0.0,
        routing_target=0.0,
        target_accuracies=(0.2, 0.3),
        select_best_checkpoint=True,
        checkpoint_epochs=(1, 2),
        selection_metric="min_target_margin",
        seed=39,
    )

    selected_index = history["epoch"].index(history["selected_epoch"])
    assert history["metadata"]["target_accuracies"] == [0.2, 0.3]
    assert len(history["min_target_margin"]) == 2
    assert history["selected_metrics"]["min_target_margin"] == max(history["min_target_margin"])
    assert np.isclose(
        history["min_target_margin"][selected_index],
        min(
            history["prefix_1_val_accuracy"][selected_index] - 0.2,
            history["prefix_2_val_accuracy"][selected_index] - 0.3,
        ),
    )


def test_case_config_uses_soft_routing_regularization():
    config = subunitary_surrogate_config_from_mapping(
        {
            "width": 16,
            "layers": 3,
            "loss_db": [2.0, 3.0],
            "classes": 10,
            "routing_limit": 7,
            "hard_routing": False,
        }
    )

    assert config.routing_limit == 7
    assert config.hard_routing is False
    assert config.input_amplitude == 1.0
    assert np.isclose(config.phase_scale, np.pi)


def test_reference_config_values_match_local_routing_notebook():
    config = subunitary_surrogate_config_from_mapping(
        {
            "width": 16,
            "layers": 5,
            "loss_db": [0.0, 1.5],
            "classes": 10,
            "routing_limit": 4,
            "hard_routing": False,
            "input_amplitude": 0.25,
            "phase_scale": np.pi,
            "readout": "temperature_softmax",
            "init_gamma": 10.0,
            "layer_name_prefix": "subunitary",
            "readout_port_start": None,
        }
    )

    assert config.layers == 5
    assert config.loss_db == (0.0, 1.5)
    assert config.routing_limit == 4
    assert config.input_amplitude == 0.25
    assert config.readout == "temperature_softmax"
    assert config.init_gamma == 10.0
    assert config.layer_name_prefix == "subunitary"
    assert config.readout_port_start is None
    assert config.nonlinearity == "repeated_phase_mask"


def test_config_accepts_bias_ports_for_constant_phase_inputs():
    config = subunitary_surrogate_config_from_mapping(
        {
            "width": 17,
            "layers": 5,
            "loss_db": [0.0, None],
            "classes": 10,
            "bias_ports": 1,
        }
    )

    assert config.width == 17
    assert config.bias_ports == 1


def test_config_accepts_input_phase_once_ablation():
    config = subunitary_surrogate_config_from_mapping(
        {
            "width": 16,
            "layers": 3,
            "loss_db": [2.0, None],
            "classes": 10,
            "nonlinearity": "input_phase_once",
        }
    )

    assert config.nonlinearity == "input_phase_once"


def test_config_accepts_minimum_loss_with_open_maximum_bound():
    config = subunitary_surrogate_config_from_mapping(
        {
            "width": 16,
            "layers": 3,
            "loss_db": [2.0, None],
            "classes": 10,
        }
    )

    assert config.loss_db == (2.0, None)


def test_hard_routed_surrogate_zeros_forbidden_transfer_entries():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=3,
        loss_db=(2.0, 3.0),
        classes=10,
        routing_limit=7,
        hard_routing=True,
    )
    model = build_subunitary_surrogate(config)
    values = jnp.ones((2, 16), dtype=jnp.float32)
    params = model.init(jax.random.key(3), values)["params"]

    matrices = extract_inverse_design_matrices(model, params, values)
    allowed = routing_mask(16, 16, 7)

    assert all(jnp.allclose(jnp.where(allowed, 0.0, matrix), 0.0) for matrix in matrices.values())
    assert all(jnp.allclose(matrix[15, 3], 0.0) for matrix in matrices.values())


def test_hard_routed_surrogate_caps_masked_spectral_norm():
    config = SubUnitarySurrogateConfig(
        width=16,
        layers=1,
        loss_db=(2.0, 3.0),
        classes=10,
        routing_limit=7,
        hard_routing=True,
    )
    model = build_subunitary_surrogate(config)
    values = jnp.ones((2, 16), dtype=jnp.float32)
    params = model.init(jax.random.key(24), values)["params"]
    matrix = next(iter(extract_inverse_design_matrices(model, params, values).values()))
    _, singular_max = insertion_loss_bounds((2.0, 3.0))

    assert float(jnp.linalg.svd(matrix, compute_uv=False)[0]) <= float(singular_max) + 1e-5


def test_extract_inverse_design_matrices_returns_one_target_per_layer():
    config = SubUnitarySurrogateConfig(width=16, layers=3, loss_db=(2.0, 3.0), classes=10)
    model = build_subunitary_surrogate(config)
    values = jnp.ones((2, 16), dtype=jnp.float32)
    variables = model.init(jax.random.key(1), values)

    matrices = extract_inverse_design_matrices(model, variables["params"], values)

    assert sorted(matrices) == ["optical_0", "optical_1", "optical_2"]
    assert all(matrix.shape == (16, 16) for matrix in matrices.values())
