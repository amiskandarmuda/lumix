from flax.core import freeze, unfreeze


def freeze_params(tree):
    return freeze(unfreeze(tree))
