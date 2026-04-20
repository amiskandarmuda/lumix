from collections.abc import Iterable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def cross_entropy(y_true: jnp.ndarray, probs: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    clipped = jnp.clip(probs, eps, 1.0)
    return -jnp.mean(jnp.sum(y_true * jnp.log(clipped), axis=-1))


def accuracy(y_true: jnp.ndarray, probs: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(y_true, axis=-1) == jnp.argmax(probs, axis=-1))


def create_state(module, rng, sample_x: jnp.ndarray, learning_rate: float) -> TrainState:
    variables = module.init(rng, sample_x)
    constants = {name: value for name, value in variables.items() if name != "params"}
    optimizer = optax.adam(learning_rate)

    def apply_fn(variable_dict, batch_x):
        return module.apply({**constants, **variable_dict}, batch_x)

    return TrainState.create(apply_fn=apply_fn, params=variables["params"], tx=optimizer)


@jax.jit
def train_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    def loss_fn(params):
        probs = state.apply_fn({"params": params}, batch_x)
        return cross_entropy(batch_y, probs), probs

    (loss_value, probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    next_state = state.apply_gradients(grads=grads)
    score = accuracy(batch_y, probs)
    return next_state, loss_value, score


@jax.jit
def eval_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    probs = state.apply_fn({"params": state.params}, batch_x)
    return cross_entropy(batch_y, probs), accuracy(batch_y, probs)


def iterate_batches(x, y, batch_size: int, rng):
    indices = jax.random.permutation(rng, x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices]


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
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    rng = jax.random.PRNGKey(seed)

    for epoch in range(1, epochs + 1):
        rng, epoch_rng = jax.random.split(rng)
        losses = []
        scores = []
        for batch_x, batch_y in iterate_batches(train_x, train_y, batch_size, epoch_rng):
            state, loss_value, score = train_step(state, batch_x, batch_y)
            losses.append(loss_value)
            scores.append(score)

        train_loss = float(jnp.mean(jnp.stack(losses)))
        train_score = float(jnp.mean(jnp.stack(scores)))
        val_loss, val_score = eval_step(state, test_x, test_y)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_score)
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_score))

    return state, history
