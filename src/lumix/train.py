import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from lumix.batching import iterate_batches
from lumix.losses import cross_entropy, cross_entropy_logits
from lumix.metrics import accuracy
from lumix.state import apply_gradients, create_state


def _run_prob_loss(params, state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    probs = state.apply_fn({"params": params}, batch_x)
    return cross_entropy(batch_y, probs), probs


def _run_logit_loss(params, state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    logits = state.apply_fn({"params": params}, batch_x)
    return cross_entropy_logits(batch_y, logits), logits


@jax.jit
def train_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    (loss_value, probs), grads = jax.value_and_grad(_run_prob_loss, has_aux=True)(state.params, state, batch_x, batch_y)
    next_state = apply_gradients(state, grads)
    score = accuracy(batch_y, probs)
    return next_state, loss_value, score


@jax.jit
def train_step_logits(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    (loss_value, logits), grads = jax.value_and_grad(_run_logit_loss, has_aux=True)(state.params, state, batch_x, batch_y)
    next_state = apply_gradients(state, grads)
    score = accuracy(batch_y, logits)
    return next_state, loss_value, score


@jax.jit
def eval_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    loss_value, probs = _run_prob_loss(state.params, state, batch_x, batch_y)
    return loss_value, accuracy(batch_y, probs)


@jax.jit
def eval_step_logits(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    loss_value, logits = _run_logit_loss(state.params, state, batch_x, batch_y)
    return loss_value, accuracy(batch_y, logits)


def _fit_loop(
    state: TrainState,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    train_step_fn,
    eval_step_fn,
):
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    rng = jax.random.PRNGKey(seed)

    for epoch in range(1, epochs + 1):
        rng, epoch_rng = jax.random.split(rng)
        losses = []
        scores = []
        for batch_x, batch_y in iterate_batches(train_x, train_y, batch_size, epoch_rng):
            state, loss_value, score = train_step_fn(state, batch_x, batch_y)
            losses.append(loss_value)
            scores.append(score)

        train_loss = float(jnp.mean(jnp.stack(losses)))
        train_score = float(jnp.mean(jnp.stack(scores)))
        val_loss, val_score = eval_step_fn(state, test_x, test_y)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_score)
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_score))

    return state, history


def fit(
    state: TrainState,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    epochs: int,
    batch_size: int,
    seed: int = 0,
):
    return _fit_loop(
        state,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs,
        batch_size,
        seed,
        train_step,
        eval_step,
    )


def fit_logits(
    state: TrainState,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    epochs: int,
    batch_size: int,
    seed: int = 0,
):
    return _fit_loop(
        state,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs,
        batch_size,
        seed,
        train_step_logits,
        eval_step_logits,
    )
