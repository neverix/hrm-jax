from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import jax.nn as jnn

# Assume flash_attn_jax is a JAX equivalent of flash_attn_func
# from flash_attn_jax import flash_attn_func

from .common import trunc_normal_init


CosSin = Tuple[jnp.ndarray, jnp.ndarray]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: jnp.ndarray):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos[None, :q.shape[1], None, :]
    sin = sin[None, :q.shape[1], None, :]
    q_embed = (q * cos.astype(q.dtype)) + (rotate_half(q) * sin.astype(q.dtype))
    k_embed = (k * cos.astype(k.dtype)) + (rotate_half(k) * sin.astype(k.dtype))
    return q_embed, k_embed


class CastedLinear(eqx.Module):
    in_features: int
    out_features: int
    use_bias: bool
    weight: jax.Array
    bias: jax.Array | None

    def __init__(self, in_features: int, out_features: int, use_bias: bool, *, key: jax.random.PRNGKey):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        wkey, bkey = jax.random.split(key)
        self.weight = trunc_normal_init(stddev=1.0 / (in_features**0.5))(
            wkey, (out_features, in_features), jnp.float32
        )
        if use_bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.weight.T.astype(x.dtype)) + (self.bias.astype(x.dtype) if self.bias is not None else 0)


class CastedEmbedding(eqx.Module):
    num_embeddings: int
    embedding_dim: int
    embedding_weight: jax.Array

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, dtype: jnp.dtype, *, key: jax.random.PRNGKey):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_weight = trunc_normal_init(stddev=init_std)(
            key, (num_embeddings, embedding_dim), dtype=dtype
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.embedding_weight[x].astype(self.embedding_weight.dtype)


class CastedSparseEmbedding(eqx.Module):
    num_embeddings: int
    embedding_dim: int
    weights: jax.Array

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, dtype: jnp.dtype, *, key: jax.random.PRNGKey):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = trunc_normal_init(stddev=init_std)(
            key, (num_embeddings, embedding_dim), dtype=dtype
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.weights[x].astype(self.weights.dtype)



class RotaryEmbedding(eqx.Module):
    dim: int
    max_position_embeddings: int
    base: float
    cos: jax.Array
    sin: jax.Array

    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos = jnp.cos(emb)
        self.sin = jnp.sin(emb)

    def __call__(self) -> CosSin:
        return self.cos, self.sin





class SwiGLU(eqx.Module):
    hidden_size: int
    expansion: float
    gate_up_proj: CastedLinear
    down_proj: CastedLinear

    def __init__(self, hidden_size: int, expansion: float, *, key: jax.random.PRNGKey):
        self.hidden_size = hidden_size
        self.expansion = expansion
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        gate_up_key, down_key = jax.random.split(key)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, use_bias=False, key=gate_up_key)
        self.down_proj = CastedLinear(inter, hidden_size, use_bias=False, key=down_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(jnn.silu(gate) * up)


class RMSNorm(eqx.Module):
    variance_epsilon: float = eqx.field(static=True)

    def __init__(self, variance_epsilon: float = 1e-6):
        self.variance_epsilon = variance_epsilon

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return hidden_states.astype(input_dtype)
