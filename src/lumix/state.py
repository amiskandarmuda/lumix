import optax
from flax.training.train_state import TrainState

from lumix.param_tree import freeze_tree, project_subunitary_params


def create_state(module, rng, sample_x, learning_rate: float) -> TrainState:
    variables = module.init(rng, sample_x)
    constants = {name: value for name, value in variables.items() if name != "params"}
    params = freeze_tree(variables["params"])
    optimizer = optax.adam(learning_rate)

    def apply_fn(variable_dict, batch_x):
        return module.apply({**constants, **variable_dict}, batch_x)

    return TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)


def apply_gradients_and_project(state: TrainState, grads) -> TrainState:
    next_state = state.apply_gradients(grads=freeze_tree(grads))
    return next_state.replace(params=project_subunitary_params(next_state.params))
