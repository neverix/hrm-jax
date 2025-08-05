from typing import Tuple, List, Dict, Optional, NamedTuple
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn
from pydantic import BaseModel

from .common import trunc_normal_init
from .layers import RMSNorm, SwiGLU, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear

import einops
from jaxtyping import Array, Float
try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
    _has_flash_attn = True
except ImportError:
    _has_flash_attn = False

def _flash_mha_forward(
    self,
    query: Float[Array, "q_seq q_size"],
    key_: Float[Array, "kv_seq k_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Array] = None,
    *,
    key: Optional[jax.random.PRNGKey] = None,
    inference: Optional[bool] = None,
    deterministic: Optional[bool] = None,
    process_heads = None,
) -> Float[Array, "q_seq o_size"]:
    del mask, key, inference, deterministic  # Unused.

    query_heads = jax.vmap(self._project, in_axes=(None, 0))(self.query_proj, query)
    key_heads = jax.vmap(self._project, in_axes=(None, 0))(self.key_proj, key_)
    value_heads = jax.vmap(self._project, in_axes=(None, 0))(self.value_proj, value)

    if process_heads is not None:
        query_heads, key_heads, value_heads = process_heads(
            query_heads, key_heads, value_heads
        )

    query_heads = einops.rearrange(query_heads, "b s h d -> b h s d")
    key_heads = einops.rearrange(key_heads, "b s h d -> b h s d")
    value_heads = einops.rearrange(value_heads, "b s h d -> b h s d")

    pad_seq_len = 128
    seq_len = query_heads.shape[2]
    padded_seq_len = ((seq_len + pad_seq_len - 1) // pad_seq_len) * pad_seq_len
    query_heads = jnp.pad(query_heads, ((0, 0), (0, 0), (0, padded_seq_len - seq_len), (0, 0)))
    key_heads = jnp.pad(key_heads, ((0, 0), (0, 0), (0, padded_seq_len - seq_len), (0, 0)))
    value_heads = jnp.pad(value_heads, ((0, 0), (0, 0), (0, padded_seq_len - value_heads.shape[2]), (0, 0)))

    attn = flash_attention(query_heads, key_heads, value_heads, causal=False)[:, :, :seq_len, :]
    attn = einops.rearrange(attn, "b h s d -> b s (h d)")

    return eqx.filter_vmap(eqx.filter_vmap(self.output_proj, in_axes=0), in_axes=0)(attn)

if _has_flash_attn:
    eqx.nn.MultiheadAttention.__call__ = _flash_mha_forward



class HierarchicalReasoningModel_ACTV1InnerCarry(NamedTuple):
    z_H: jnp.ndarray
    z_L: jnp.ndarray


class HierarchicalReasoningModel_ACTV1Carry(NamedTuple):
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: jnp.ndarray
    halted: jnp.ndarray
    current_data: Dict[str, jnp.ndarray]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(eqx.Module):
    attn: nn.MultiheadAttention
    mlp: SwiGLU
    norm1: RMSNorm
    norm2: RMSNorm

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, *, key: jax.random.PRNGKey):
        attn_key, mlp_key = jax.random.split(key)
        self.attn = nn.MultiheadAttention(
            num_heads=config.num_heads,
            query_size=config.hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=attn_key,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion, key=mlp_key)
        self.norm1 = RMSNorm(variance_epsilon=config.rms_norm_eps)
        self.norm2 = RMSNorm(variance_epsilon=config.rms_norm_eps)

    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray) -> jnp.ndarray:
        normed_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(query=normed_hidden_states, key_=normed_hidden_states, value=normed_hidden_states)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(eqx.Module):
    layers: List[HierarchicalReasoningModel_ACTV1Block]

    def __init__(self, num_layers: int, config: HierarchicalReasoningModel_ACTV1Config, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, num_layers)
        self.layers = [
            HierarchicalReasoningModel_ACTV1Block(config, key=k) for k in keys
        ]

    def __call__(
        self, hidden_states: jnp.ndarray, input_injection: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(eqx.Module):
    config: HierarchicalReasoningModel_ACTV1Config = eqx.field(static=True)
    embed_tokens: CastedEmbedding
    lm_head: CastedLinear
    q_head: CastedLinear
    puzzle_emb: Optional[CastedEmbedding]
    rotary_emb: Optional[RotaryEmbedding]
    embed_pos: Optional[CastedEmbedding]
    H_level: HierarchicalReasoningModel_ACTV1ReasoningModule
    L_level: HierarchicalReasoningModel_ACTV1ReasoningModule

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, *, key: jax.random.PRNGKey):
        self.config = config
        keys = jax.random.split(key, 6)
        forward_dtype = getattr(jnp, config.forward_dtype)
        embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size, init_std=embed_init_std, dtype=forward_dtype, key=keys[0]
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, use_bias=False, key=keys[1])
        self.q_head = CastedLinear(config.hidden_size, 2, use_bias=True, key=keys[2])

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                init_std=0,
                dtype=forward_dtype,
                key=keys[3],
            )
        else:
            self.puzzle_emb = None

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
            self.embed_pos = None
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                dtype=forward_dtype,
                key=keys[4],
            )
            self.rotary_emb = None
        else:
            raise NotImplementedError()

        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            config.H_layers, config, key=keys[5]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            config.L_layers, config, key=keys[5]
        )

    @property
    def puzzle_emb_len(self):
        return -(self.config.puzzle_emb_ndim // -self.config.hidden_size)

    def _input_embeddings(self, input: jnp.ndarray, puzzle_identifiers: jnp.ndarray):
        embedding = self.embed_tokens(input.astype(jnp.int32))
        if self.puzzle_emb is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = jnp.pad(puzzle_embedding, ((0, 0), (0, pad_count)))
            embedding = jnp.concatenate(
                (
                    puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.config.hidden_size),
                    embedding,
                ),
                axis=-2,
            )
        if self.embed_pos is not None:
            embedding = 0.707106781 * (embedding + self.embed_pos.weight.astype(getattr(jnp, self.config.forward_dtype)))
        return math.sqrt(self.config.hidden_size) * embedding

    def __call__(
        self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        seq_info = dict(cos_sin=self.rotary_emb() if self.rotary_emb is not None else None)
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        def fwd_iter(z_H, z_L):
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)
            return z_H, z_L

        z_H, z_L = jax.lax.stop_gradient(fwd_iter(carry.z_H, carry.z_L))
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=jax.lax.stop_gradient(z_H), z_L=jax.lax.stop_gradient(z_L))
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).astype(jnp.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(eqx.Module):
    config: HierarchicalReasoningModel_ACTV1Config = eqx.field(static=True)
    inner: HierarchicalReasoningModel_ACTV1_Inner
    H_init: jax.Array
    L_init: jax.Array

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, *, key: jax.random.PRNGKey):
        self.config = config
        inner_key, h_key, l_key = jax.random.split(key, 3)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(config, key=inner_key)
        self.H_init = trunc_normal_init(stddev=1.0)(
            h_key, (config.hidden_size,), getattr(jnp, config.forward_dtype)
        )
        self.L_init = trunc_normal_init(stddev=1.0)(
            l_key, (config.hidden_size,), getattr(jnp, config.forward_dtype)
        )

    def __call__(
        self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, jnp.ndarray], *, key: jax.random.PRNGKey
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, jnp.ndarray]]:
        def reset_carry(reset_flag: jnp.ndarray, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
            return HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=jnp.where(reset_flag.reshape(-1, 1, 1), self.H_init, carry.z_H),
                z_L=jnp.where(reset_flag.reshape(-1, 1, 1), self.L_init, carry.z_L),
            )

        new_inner_carry = reset_carry(carry.halted, carry.inner_carry)
        new_steps = jnp.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: jnp.where(
                carry.halted.reshape((-1,) + (1,) * (v.ndim - 1)), v, carry.current_data.get(k, v)
            )
            for k, v in batch.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        halted = is_last_step

        if self.config.halt_max_steps > 1:
            halted = halted | (q_halt_logits > q_continue_logits)
            min_halt_steps = (
                (jax.random.uniform(key) < self.config.halt_exploration_prob)
                * jax.random.randint(
                    key,
                    new_steps.shape,
                    2,
                    self.config.halt_max_steps + 1,
                    new_steps.dtype,
                )
            )
            halted = halted & (new_steps >= min_halt_steps)
            next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
            outputs["target_q_continue"] = jax.nn.sigmoid(
                jnp.where(
                    is_last_step,
                    next_q_halt_logits,
                    jnp.maximum(next_q_halt_logits, next_q_continue_logits),
                )
            )

        return (
            HierarchicalReasoningModel_ACTV1Carry(
                new_inner_carry, new_steps, halted, new_current_data
            ),
            outputs,
        )

if __name__ == "__main__":
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=4,
        seq_len=128,
        puzzle_emb_ndim=128,
        num_puzzle_identifiers=10,
        vocab_size=1000,
        H_cycles=2,
        L_cycles=2,
        H_layers=2,
        L_layers=2,
        hidden_size=256,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=10,
        halt_exploration_prob=0.1,
    )

    key = jax.random.PRNGKey(0)
    model = HierarchicalReasoningModel_ACTV1(config, key=key)

    @jax.jit
    def init_and_run_model():
        key = jax.random.PRNGKey(0)
        batch = {
            "inputs": jnp.ones((config.batch_size, config.seq_len), dtype=jnp.int32),
            "puzzle_identifiers": jnp.ones((config.batch_size,), dtype=jnp.int32),
        }
        
        puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=jnp.zeros((config.batch_size, config.seq_len + puzzle_emb_len, config.hidden_size)),
            z_L=jnp.zeros((config.batch_size, config.seq_len + puzzle_emb_len, config.hidden_size)),
        )
        
        carry = HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,
            steps=jnp.zeros((config.batch_size,), dtype=jnp.int32),
            halted=jnp.ones((config.batch_size,), dtype=jnp.bool_),
            current_data={k: jnp.zeros_like(v) for k, v in batch.items()},
        )
        
        (new_carry, outputs) = model(carry, batch, key=key)
        return new_carry, outputs

    new_carry, outputs = init_and_run_model()

    print("Model created and applied successfully.")
    print("Output logits shape:", outputs["logits"].shape)
    print("Halted shape:", new_carry.halted.shape)
