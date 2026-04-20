from flax.core import freeze, unfreeze


def freeze_tree(tree):
    return freeze(unfreeze(tree))
