from typing import Tuple, List, Dict, Optional, NamedTuple
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from pydantic import BaseModel

from .common import trunc_normal_init
from .layers import RMSNorm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear


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


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    config: HierarchicalReasoningModel_ACTV1Config

    @nn.compact
    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray) -> jnp.ndarray:
        # Post Norm
        # Self Attention
        attn = Attention(
            hidden_size=self.config.hidden_size,
            head_dim=self.config.hidden_size // self.config.num_heads,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
            causal=False,
        )
        hidden_states = RMSNorm(variance_epsilon=self.config.rms_norm_eps)(
            hidden_states + attn(cos_sin=cos_sin, hidden_states=hidden_states)
        )
        # Fully Connected
        mlp = SwiGLU(hidden_size=self.config.hidden_size, expansion=self.config.expansion)
        hidden_states = RMSNorm(variance_epsilon=self.config.rms_norm_eps)(
            hidden_states + mlp(hidden_states)
        )
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    layers: List[nn.Module]

    @nn.compact
    def __call__(
        self, hidden_states: jnp.ndarray, input_injection: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    config: HierarchicalReasoningModel_ACTV1Config

    @nn.compact
    def __call__(
        self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        forward_dtype = getattr(jnp, self.config.forward_dtype)
        embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / embed_scale
        puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)

        embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, dtype=forward_dtype
        )
        lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, use_bias=False)
        q_head = CastedLinear(self.config.hidden_size, 2, use_bias=True)

        if self.config.puzzle_emb_ndim > 0:
            puzzle_emb = CastedEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                init_std=0,
                dtype=forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            embed_pos = CastedEmbedding(
                self.config.seq_len + puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                dtype=forward_dtype,
            )
        else:
            raise NotImplementedError()

        H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.H_layers)
            ]
        )
        L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.L_layers)
            ]
        )

        def _input_embeddings(input: jnp.ndarray, puzzle_identifiers: jnp.ndarray):
            embedding = embed_tokens(input.astype(jnp.int32))
            if self.config.puzzle_emb_ndim > 0:
                puzzle_embedding = puzzle_emb(puzzle_identifiers)
                pad_count = puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = jnp.pad(puzzle_embedding, ((0, 0), (0, pad_count)))
                embedding = jnp.concatenate(
                    (
                        puzzle_embedding.reshape(-1, puzzle_emb_len, self.config.hidden_size),
                        embedding,
                    ),
                    axis=-2,
                )
            if self.config.pos_encodings == "learned":
                embedding = 0.707106781 * (embedding + embed_pos.embedding_weight.astype(forward_dtype))
            return embed_scale * embedding

        seq_info = dict(cos_sin=rotary_emb() if hasattr(self, "rotary_emb") else None)
        input_embeddings = _input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        def fwd_iter(z_H, z_L):
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = L_level(z_L, z_H + input_embeddings, **seq_info)
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = H_level(z_H, z_L, **seq_info)
            return z_H, z_L

        z_H, z_L = jax.lax.stop_gradient(fwd_iter(carry.z_H, carry.z_L))
        z_L = L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = H_level(z_H, z_L, **seq_info)

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=jax.lax.stop_gradient(z_H), z_L=jax.lax.stop_gradient(z_L))
        output = lm_head(z_H)[:, puzzle_emb_len:]
        q_logits = q_head(z_H[:, 0]).astype(jnp.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    config: HierarchicalReasoningModel_ACTV1Config

    @nn.compact
    def __call__(
        self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, jnp.ndarray]]:
        inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)
        
        H_init = self.param("H_init", trunc_normal_init(stddev=1.0), (self.config.hidden_size,), getattr(jnp, self.config.forward_dtype))
        L_init = self.param("L_init", trunc_normal_init(stddev=1.0), (self.config.hidden_size,), getattr(jnp, self.config.forward_dtype))

        def reset_carry(reset_flag: jnp.ndarray, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
            return HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=jnp.where(reset_flag.reshape(-1, 1, 1), H_init, carry.z_H),
                z_L=jnp.where(reset_flag.reshape(-1, 1, 1), L_init, carry.z_L),
            )

        new_inner_carry = reset_carry(carry.halted, carry.inner_carry)
        new_steps = jnp.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: jnp.where(
                carry.halted.reshape((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = inner(
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
                jax.random.uniform(self.make_rng("dropout")) < self.config.halt_exploration_prob
            ) * jax.random.randint(
                self.make_rng("dropout"),
                new_steps.shape,
                2,
                self.config.halt_max_steps + 1,
                new_steps.dtype,
            )
            halted = halted & (new_steps >= min_halt_steps)
            next_q_halt_logits, next_q_continue_logits = inner(new_inner_carry, new_current_data)[-1]
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

    model = HierarchicalReasoningModel_ACTV1(config)

    @jax.jit
    def init_model():
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
        
        params = model.init({"params": key, "dropout": key}, carry, batch)
        return params, carry, batch

    params, carry, batch = init_model()
    key = jax.random.PRNGKey(42)
    (new_carry, outputs), updated_state = model.apply(params, carry, batch, mutable=["batch_stats"], rngs={"dropout": key})

    print("Model created and applied successfully.")
    print("Output logits shape:", outputs["logits"].shape)
    print("Halted shape:", new_carry.halted.shape)
