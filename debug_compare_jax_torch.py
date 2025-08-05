import argparse
import os
from typing import Dict
from collections import OrderedDict

import yaml
import torch
import jax
import jax.numpy as jnp
import equinox as eqx

# PyTorch / JAX model imports
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1 as HRM_PT
from models.hrm_jax.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1 as HRM_JAX,
    HierarchicalReasoningModel_ACTV1Config as HRM_JAX_Config,
    HierarchicalReasoningModel_ACTV1InnerCarry as InnerCarry_JAX,
    HierarchicalReasoningModel_ACTV1Carry as Carry_JAX,
)
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Carry as Carry_PT,
    HierarchicalReasoningModel_ACTV1InnerCarry as InnerCarry_PT,
)
from pretrain_jax import get_getter


def load_arch_config(config_path: str) -> Dict:
    """Load architecture section from YAML config. If the YAML cannot be parsed
    (e.g. malformed), fall back to a hard-coded minimal subset that matches the
    released Maze checkpoint."""
    arch_defaults = {
        "H_cycles": 2,
        "H_layers": 4,
        "L_cycles": 2,
        "L_layers": 4,
        "expansion": 4,
        "halt_exploration_prob": 0.1,
        "halt_max_steps": 16,
        "hidden_size": 512,
        "num_heads": 8,
        "pos_encodings": "rope",
        "puzzle_emb_ndim": 512,
    }

    if not os.path.exists(config_path):
        return arch_defaults

    try:
        with open(config_path, "rt") as f:
            cfg_all = yaml.safe_load(f)
        if cfg_all and "arch" in cfg_all:
            arch_cfg = cfg_all["arch"]
            arch_defaults.update({k: v for k, v in arch_cfg.items() if not isinstance(v, dict)})
    except Exception:
        # YAML parsing failed – use defaults
        pass
    return arch_defaults


def torch_load_state_dict(ckpt_path: str):
    """Load PyTorch checkpoint and return cleaned state_dict."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # Remove DataParallel/Distributed prefixes
    cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
    return cleaned


def build_config_from_state_dict(arch_cfg: Dict, state_dict: Dict) -> Dict:
    """Derive missing fields (vocab_size, num_puzzle_identifiers) from the
    checkpoint weights so that the model config perfectly matches the saved
    parameters."""
    vocab_size = None
    num_puzzle_ids = 1  # default when no puzzle embedding is used

    for k, v in state_dict.items():
        if "embed_tokens.embedding_weight" in k:
            vocab_size = v.shape[0]
        elif "puzzle_emb.embedding_weight" in k:
            num_puzzle_ids = v.shape[0]
    if vocab_size is None:
        raise ValueError("Unable to infer vocab_size from checkpoint.")

    # Select an arbitrary (but consistent with config) seq_len for the random
    # input. 32 works for small debug runs.
    seq_len = 127

    cfg = {
        "batch_size": 1,
        "seq_len": seq_len,
        "puzzle_emb_ndim": arch_cfg["puzzle_emb_ndim"],
        "num_puzzle_identifiers": num_puzzle_ids,
        "vocab_size": vocab_size,
        **{k: arch_cfg[k] for k in [
            "H_cycles",
            "L_cycles",
            "H_layers",
            "L_layers",
            "hidden_size",
            "expansion",
            "num_heads",
            "pos_encodings",
            "halt_max_steps",
            "halt_exploration_prob",
        ]},
    }
    return cfg


def convert_to_jax(model_jax: eqx.Module, torch_sd: Dict):
    """Port weights from PyTorch state_dict to the JAX model in-place (returns a
    new EQX tree)."""
    new_sd = OrderedDict()
    for k, v in torch_sd.items():
        v = v.float()
        new_sd[k] = jnp.asarray(v.numpy())

    for k, v in new_sd.items():
        name = k
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod."):]
        if name.startswith("model."):
            name = name[len("model."):]
        getter = get_getter(name)
        model_jax = eqx.tree_at(getter, model_jax, jnp.asarray(v))
    return model_jax


def build_random_batch_pt(cfg: Dict):
    inputs = torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), dtype=torch.int32)
    puzzle_ids = torch.randint(0, cfg["num_puzzle_identifiers"], (cfg["batch_size"],), dtype=torch.int32)
    return {"inputs": inputs, "puzzle_identifiers": puzzle_ids}


def build_random_batch_jax(batch_pt):
    return {k: jnp.asarray(v.cpu().numpy()) for k, v in batch_pt.items()}

def build_zero_batch_jax(batch):
    return {k: jnp.zeros_like(v) for k, v in batch.items()}

def build_zero_batch_pt(cfg: Dict):
    """Return a zero-filled batch matching shapes expected by the model."""
    return {
        "inputs": torch.zeros((cfg["batch_size"], cfg["seq_len"]), dtype=torch.int32),
        "puzzle_identifiers": torch.zeros((cfg["batch_size"],), dtype=torch.int32),
    }


def create_pt_carry(cfg: Dict):
    puzzle_emb_len = -(cfg["puzzle_emb_ndim"] // -cfg["hidden_size"])
    z_shape = (cfg["batch_size"], cfg["seq_len"] + puzzle_emb_len, cfg["hidden_size"])

    inner_carry = InnerCarry_PT(
        z_H=torch.zeros(z_shape, dtype=torch.float32),
        z_L=torch.zeros(z_shape, dtype=torch.float32),
    )

    zero_batch = build_zero_batch_pt(cfg)

    return Carry_PT(
        inner_carry=inner_carry,
        steps=torch.zeros((cfg["batch_size"],), dtype=torch.int32),
        halted=torch.ones((cfg["batch_size"],), dtype=torch.bool),
        current_data=zero_batch,
    )


def create_jax_carry(cfg: Dict,):
    puzzle_emb_len = -(cfg["puzzle_emb_ndim"] // -cfg["hidden_size"])
    z_shape = (cfg["batch_size"], cfg["seq_len"] + puzzle_emb_len, cfg["hidden_size"])
    inner_carry = InnerCarry_JAX(
        z_H=jnp.zeros(z_shape, dtype=jnp.float32),
        z_L=jnp.zeros(z_shape, dtype=jnp.float32),
    )
    return Carry_JAX(
        inner_carry=inner_carry,
        steps=jnp.zeros((cfg["batch_size"],), dtype=jnp.int32),
        halted=jnp.ones((cfg["batch_size"],), dtype=jnp.bool_),
        current_data=build_zero_batch_jax({"inputs": jnp.zeros((cfg["batch_size"], cfg["seq_len"]), dtype=jnp.int32), "puzzle_identifiers": jnp.zeros((cfg["batch_size"],), dtype=jnp.int32)}),
    )


def main():
    parser = argparse.ArgumentParser(description="Debug equivalence between PyTorch and JAX checkpoints")
    parser.add_argument("--ckpt", default="eval_data/maze/checkpoint.pth", help="Path to .pth checkpoint")
    parser.add_argument("--config", default="eval_data/maze/all_config.yaml", help="Path to YAML config")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for output comparison")
    # Behaviour flags (defaults: ACT disabled, compare blocks enabled)
    parser.add_argument("--enable-act", action="store_true", help="Enable ACT (keeps original halt_max_steps/halt_exploration_prob)")
    parser.add_argument("--no-compare-blocks", action="store_true", help="Skip per-block/MLP/attn comparisons")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load checkpoint + derive config
    # ---------------------------------------------------------------------
    arch_cfg = load_arch_config(args.config)
    if not args.enable_act:
        arch_cfg["halt_max_steps"] = 1
        arch_cfg["halt_exploration_prob"] = 0.0
    sd_pt = torch_load_state_dict(args.ckpt)
    cfg = build_config_from_state_dict(arch_cfg, sd_pt)
    if not args.enable_act:
        cfg["halt_max_steps"] = 1
        cfg["halt_exploration_prob"] = 0.0

    # -----------------------------------------------------
    # Optional fine-grained block comparison helper
    # -----------------------------------------------------

    def _tensor_to_np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else jax.device_get(t)

    def compare_block(pt_block, jax_block, name: str):
        bs, seq_len, dim = cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"]
        x_pt = torch.randn(bs, seq_len, dim, dtype=torch.float32)
        with torch.no_grad():
            out_pt = pt_block(cos_sin=None, hidden_states=x_pt).cpu()

        x_jax = jnp.asarray(x_pt.numpy())
        out_jax = jax_block(cos_sin=None, hidden_states=x_jax)
        diff = abs(_tensor_to_np(out_pt) - _tensor_to_np(out_jax)).max()
        print(f"[Block diff] {name}: {diff}")
        return diff

    # ---------------------------------------------------------------------
    # Build PyTorch model + load weights
    # ---------------------------------------------------------------------
    model_pt = HRM_PT(cfg).eval()
    sd_pt = {k[len("_orig_mod.model."):]: v for k, v in sd_pt.items()}
    missing, unexpected = model_pt.load_state_dict(sd_pt, strict=False)
    if missing:
        print("[PyTorch] Missing keys:", missing)
    if unexpected:
        print("[PyTorch] Unexpected keys:", unexpected)

    batch_pt = build_random_batch_pt(cfg)
    carry_pt = create_pt_carry(cfg)
    with torch.no_grad():
        new_carry_pt, outputs_pt = model_pt(carry_pt, batch_pt)
    logits_pt = outputs_pt["logits"].cpu().float().numpy()

    # ---------------------------------------------------------------------
    # Build JAX model + load weights
    # ---------------------------------------------------------------------
    key = jax.random.PRNGKey(0)
    model_jax = HRM_JAX(HRM_JAX_Config(**cfg), key=key)
    model_jax = convert_to_jax(model_jax, sd_pt)

    batch_jax = build_random_batch_jax(batch_pt)
    carry_jax = create_jax_carry(cfg)
    key, subkey = jax.random.split(key)
    new_carry_jax, outputs_jax = model_jax(carry_jax, batch_jax, key=subkey)
    logits_jax = jax.device_get(outputs_jax["logits"]).astype("float32")

    # ---------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------
    diff_logits = abs(logits_pt - logits_jax).mean()

    # ----------------------------------
    # Compare inner state tensors (z_H, z_L)
    # ----------------------------------

    zH_pt_before = carry_pt.inner_carry.z_H.detach().cpu().numpy()
    zH_jax_before = jax.device_get(carry_jax.inner_carry.z_H).astype("float32")
    init_zH_diff = abs(zH_pt_before - zH_jax_before).max()

    zL_pt_before = carry_pt.inner_carry.z_L.detach().cpu().numpy()
    zL_jax_before = jax.device_get(carry_jax.inner_carry.z_L).astype("float32")
    init_zL_diff = abs(zL_pt_before - zL_jax_before).max()

    zH_pt_after = new_carry_pt.inner_carry.z_H.detach().cpu().numpy()
    zH_jax_after = jax.device_get(new_carry_jax.inner_carry.z_H).astype("float32")
    final_zH_diff = abs(zH_pt_after - zH_jax_after).max()

    zL_pt_after = new_carry_pt.inner_carry.z_L.detach().cpu().numpy()
    zL_jax_after = jax.device_get(new_carry_jax.inner_carry.z_L).astype("float32")
    final_zL_diff = abs(zL_pt_after - zL_jax_after).max()

    # Print results
    print("Max absolute logit difference:", diff_logits)
    print("Initial z_H diff:", init_zH_diff, "| Initial z_L diff:", init_zL_diff)
    print("Final   z_H diff:", final_zH_diff, "| Final   z_L diff:", final_zL_diff)

    if diff_logits < args.atol and final_zH_diff < args.atol and final_zL_diff < args.atol:
        print("SUCCESS: Outputs and inner states match within tolerance.")
    else:
        print("WARNING: Mismatch detected beyond tolerance.")

    # -----------------------------------------------------
    # Fine-grained block comparison (optional)
    # -----------------------------------------------------

    if not args.no_compare_blocks:
        print("\n--- Per-block comparison (random input, converted weights) ---")

        # H-level blocks
        for idx, (pt_b, j_b) in enumerate(zip(model_pt.inner.H_level.layers, model_jax.inner.H_level.layers)):
            diff_block = compare_block(pt_b, j_b, f"H_level.block[{idx}]")
            # MLP diff
            x_pt = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"], dtype=torch.float32)
            with torch.no_grad():
                out_pt_mlp = pt_b.mlp(x_pt).cpu()
            out_jax_mlp = j_b.mlp(jnp.asarray(x_pt.numpy()))
            diff_mlp = abs(_tensor_to_np(out_pt_mlp) - _tensor_to_np(out_jax_mlp)).max()
            print(f"   └─ MLP diff: {diff_mlp}")

            # Attention diff
            if hasattr(pt_b, "self_attn") and hasattr(j_b, "self_attn"):
                with torch.no_grad():
                    out_pt_attn = pt_b.self_attn(cos_sin=None, hidden_states=x_pt).cpu()
                out_jax_attn = j_b.self_attn(cos_sin=None, hidden_states=jnp.asarray(x_pt.numpy()))
                diff_attn = abs(_tensor_to_np(out_pt_attn) - _tensor_to_np(out_jax_attn)).max()
                print(f"   └─ Attn diff: {diff_attn}")

        # L-level blocks
        for idx, (pt_b, j_b) in enumerate(zip(model_pt.inner.L_level.layers, model_jax.inner.L_level.layers)):
            diff_block = compare_block(pt_b, j_b, f"L_level.block[{idx}]")
            x_pt = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"], dtype=torch.float32)
            with torch.no_grad():
                out_pt_mlp = pt_b.mlp(x_pt).cpu()
            out_jax_mlp = j_b.mlp(jnp.asarray(x_pt.numpy()))
            diff_mlp = abs(_tensor_to_np(out_pt_mlp) - _tensor_to_np(out_jax_mlp)).max()
            print(f"   └─ MLP diff: {diff_mlp}")

            if hasattr(pt_b, "self_attn") and hasattr(j_b, "self_attn"):
                with torch.no_grad():
                    out_pt_attn = pt_b.self_attn(cos_sin=None, hidden_states=x_pt).cpu()
                out_jax_attn = j_b.self_attn(cos_sin=None, hidden_states=jnp.asarray(x_pt.numpy()))
                diff_attn = abs(_tensor_to_np(out_pt_attn) - _tensor_to_np(out_jax_attn)).max()
                print(f"   └─ Attn diff: {diff_attn}")


if __name__ == "__main__":
    main() 