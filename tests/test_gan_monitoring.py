import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from GAN.monitor import MnistMonitorClassifier, StaleTracker, evaluate_monitor, load_or_train_classifier
from GAN.training import TrainingConfig, train_wgan


def test_monitor_classifier_outputs_ten_class_logits():
    model = MnistMonitorClassifier()
    images = jnp.ones((4, 28, 28), dtype=jnp.float32)

    variables = model.init(jax.random.key(0), images)
    logits = model.apply(variables, images)

    assert logits.shape == (4, 10)


def test_monitor_metrics_include_stale_flags(tmp_path: Path):
    images = jnp.ones((16, 28, 28), dtype=jnp.float32) * 0.5
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(16) % 10]
    classifier_state = load_or_train_classifier(
        images,
        labels,
        checkpoint_path=tmp_path / "classifier.msgpack",
        rng=jax.random.key(1),
        epochs=1,
        batch_size=8,
    )

    metrics = evaluate_monitor(classifier_state, images, labels, StaleTracker())

    assert "monitor/label_agreement" in metrics
    assert "monitor/mean_confidence" in metrics
    assert "monitor/stale_low_agreement" in metrics
    assert "monitor/stale_low_diversity" in metrics
    assert "monitor/stale_flat_generator" in metrics


def test_training_smoke_writes_metrics_samples_and_checkpoints(tmp_path: Path):
    result = train_wgan(
        root_dir=tmp_path / "GAN",
        config=TrainingConfig(
            run_name="pytest-smoke",
            batch_size=8,
            epochs=1,
            n_critic=2,
            classifier_epochs=1,
            classifier_batch_size=8,
            train_limit=32,
            test_limit=16,
            sample_repeats_per_class=2,
            monitor_interval=1,
            sample_interval=1,
            checkpoint_interval=1,
            log_interval=1,
            max_generator_steps=1,
        ),
    )

    run_paths = result["run_paths"]
    metrics_path = run_paths["metrics_path"]
    sample_files = list(run_paths["samples_dir"].glob("*.pgm"))
    checkpoint_files = list(run_paths["checkpoints_dir"].glob("*.msgpack"))

    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert "train/critic_loss" in payload
    assert "monitor/label_agreement" in payload
    assert "monitor/stale_low_agreement" in payload
    assert sample_files
    assert checkpoint_files
