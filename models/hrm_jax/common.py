import jax

def trunc_normal_init(stddev=1.0, lower=-2.0, upper=2.0):
    return lambda key, shape, dtype: jax.random.truncated_normal(key, lower, upper, shape, dtype) * stddev
