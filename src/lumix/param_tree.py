from flax.core import freeze, unfreeze

import jax.numpy as jnp

from lumix.functional.subunitary import project_subunitary_to_bounds


def freeze_tree(tree):
    return freeze(unfreeze(tree))


def project_subunitary_params(params):
    mutable = unfreeze(params)

    def visit(node):
        if not isinstance(node, dict):
            return
        if {"raw_re", "raw_im", "singular_min", "singular_max"} <= set(node):
            raw = node["raw_re"] + 1j * node["raw_im"]
            projected = project_subunitary_to_bounds(
                raw,
                node["singular_min"],
                node["singular_max"],
            )
            node["raw_re"] = jnp.real(projected)
            node["raw_im"] = jnp.imag(projected)
        for value in node.values():
            visit(value)

    visit(mutable)
    return freeze(mutable)
