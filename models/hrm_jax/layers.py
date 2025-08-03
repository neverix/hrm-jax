from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

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
    q_embed = (q * cos.astype(q.dtype)) + (rotate_half(q) * sin.astype(q.dtype))
    k_embed = (k * cos.astype(k.dtype)) + (rotate_half(k) * sin.astype(k.dtype))
    return q_embed, k_embed


class CastedLinear(nn.Module):
    in_features: int
    out_features: int
    use_bias: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            trunc_normal_init(stddev=1.0 / (self.in_features**0.5)),
            (self.out_features, self.in_features),
            x.dtype,
        )
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.out_features,))
        else:
            bias = None
        return jnp.dot(x, kernel.T.astype(x.dtype)) + (bias.astype(x.dtype) if bias is not None else 0)


class CastedEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    init_std: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        embedding = self.param(
            "embedding",
            trunc_normal_init(stddev=self.init_std),
            (self.num_embeddings, self.embedding_dim),
            dtype=self.dtype,
        )
        return embedding[x].astype(self.dtype)


class RotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int
    base: float

    @nn.compact
    def __call__(self):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


class Attention(nn.Module):
    hidden_size: int
    head_dim: int
    num_heads: int
    num_key_value_heads: int
    causal: bool = False

    @nn.compact
    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = hidden_states.shape

        qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            use_bias=False,
        )
        o_proj = CastedLinear(self.head_dim * self.num_heads, self.hidden_size, use_bias=False)

        qkv = qkv_proj(hidden_states)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        # attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        # attn_output = attn_output[0]
        # For now, using standard dot-product attention
        attn_weights = nn.dot_product_attention(query, key, value, deterministic=True, dropout_rate=0.0)
        attn_output = attn_weights.reshape(batch_size, seq_len, self.head_dim * self.num_heads)


        return o_proj(attn_output)


class SwiGLU(nn.Module):
    hidden_size: int
    expansion: float

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inter = _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
        gate_up_proj = CastedLinear(self.hidden_size, inter * 2, use_bias=False)
        down_proj = CastedLinear(inter, self.hidden_size, use_bias=False)

        gate, up = jnp.split(gate_up_proj(x), 2, axis=-1)
        return down_proj(nn.silu(gate) * up)


class RMSNorm(nn.Module):
    variance_epsilon: float = 1e-6

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return hidden_states.astype(input_dtype)
