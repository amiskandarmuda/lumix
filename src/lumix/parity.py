import time

import jax
import jax.numpy as jnp


def benchmark_forward(function, values: jnp.ndarray, iterations: int = 50) -> float:
    compiled = jax.jit(function)
    compiled(values).block_until_ready()
    start = time.perf_counter()
    for _ in range(iterations):
        result = compiled(values)
    result.block_until_ready()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iterations
