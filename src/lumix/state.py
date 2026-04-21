import optax
from flax.training.train_state import TrainState

from lumix.params import freeze_params


def create_state(module, rng, sample_x, learning_rate: float) -> TrainState:
    variables = module.init(rng, sample_x)
    constants = {name: value for name, value in variables.items() if name != "params"}
    params = freeze_params(variables["params"])
    optimizer = optax.adam(learning_rate)

    def apply_fn(variable_dict, batch_x):
        return module.apply({**constants, **variable_dict}, batch_x)

    return TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)


def apply_gradients(state: TrainState, grads) -> TrainState:
    return state.apply_gradients(grads=freeze_params(grads))
