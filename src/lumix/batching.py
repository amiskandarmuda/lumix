import jax


def iterate_batches(x, y, batch_size: int, rng):
    indices = jax.random.permutation(rng, x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices]
