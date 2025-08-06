import argparse
import os
from typing import Dict, Any, Tuple

import numpy as np
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import tqdm

# Data loading helpers
from pretrain import PretrainConfig, create_dataloader

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1 as HRM_PT
from models.losses import ACTLossHead as ACTLossHead_PT
from models.hrm_jax.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1 as HRM_JAX,
    HierarchicalReasoningModel_ACTV1Config as HRM_JAX_Config,
)

from debug_compare_jax_torch import (
    load_arch_config,
    torch_load_state_dict,
    build_config_from_state_dict,
    convert_to_jax,
    build_random_batch_pt,
)

from models.hrm_jax import losses as jax_losses


# Helper to convert arrays to numpy
def _tensor_to_np(x: Any):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.detach().cpu().numpy()
    try:
        import numpy as _np
        import jax as _jax
        if isinstance(x, _jax.Array):
            return _np.asarray(x)
    except ImportError:  # running only CPU / torch
        pass
    return x

def max_abs_diff(a: Any, b: Any) -> float:
    """Compute maximum absolute difference between two (possibly nested) structures."""
    a, b = _tensor_to_np(a), _tensor_to_np(b)
    a, b = a.astype("float32"), b.astype("float32")
    return np.max(np.abs(a - b))  # type: ignore


def step_compare(pt_model, jax_model, batch_pt: Dict[str, torch.Tensor], atol: float):
    # Build carries
    carry_pt = pt_model.initial_carry(batch_pt)
    batch_jax = {k: jnp.asarray(v.cpu().numpy()) for k, v in batch_pt.items()}
    carry_jax = jax_model.initial_carry(batch_jax)
    
    for k in ["inner_carry", "steps", "halted"]:
        v = getattr(carry_pt, k)
        v2 = getattr(carry_jax, k)
        if k == "inner_carry":
            for k2 in ["z_H", "z_L"]:
                v1 = getattr(v, k2)
                v2_ = getattr(v2, k2)
                v1, v2_ = _tensor_to_np(v1), _tensor_to_np(v2_)
                print(f"{k}.{k2}: {v1.shape}")
                print(f"{k}.{k2}: {v2_.shape}")
                print(f"diff: {jnp.max(jnp.abs(v1 - v2_))}")
                print(f"mean: {jnp.mean(jnp.abs(v1))}")
        else:
            v, v2 = _tensor_to_np(v), _tensor_to_np(v2)
            print(f"{k}: {v.shape}")
            print(f"{k}: {v2.shape}")
            v, v2 = v.astype("float32"), v2.astype("float32")
            print(f"diff: {jnp.max(jnp.abs(v - v2))}")
            print(f"mean: {jnp.mean(jnp.abs(v))}")
            print()

    key = jax.random.PRNGKey(0)

    @eqx.filter_jit
    def jax_step(m, c, b, k):
        return m(return_keys=[], carry=c, batch=b, key=k)

    step = 0
    inf_model = eqx.nn.inference_mode(jax_model)
    pt_model.eval()
    while True:
        # JAX step
        key, subkey = jax.random.split(key)
        carry_jax, loss_jax, metrics_jax, _, all_finish_jax = jax_step(inf_model, carry_jax, batch_jax, subkey)

        # PyTorch step
        with torch.no_grad():
            carry_pt, loss_pt, metrics_pt, _, all_finish_pt = pt_model(carry=carry_pt, batch=batch_pt, return_keys=[])

        # Compare losses
        diff_loss = abs(loss_pt.item() - float(loss_jax))
        # Compare carries (inner z_H, z_L)
        diff_zH = max_abs_diff(carry_pt.inner_carry.z_H, carry_jax.inner_carry.z_H)
        diff_zL = max_abs_diff(carry_pt.inner_carry.z_L, carry_jax.inner_carry.z_L)
        diff_steps = max_abs_diff(carry_pt.steps, carry_jax.steps)
        diff_halted = max_abs_diff(carry_pt.halted, carry_jax.halted)

        print(f"Step {step}: loss diff={diff_loss:.4e}, z_H diff={diff_zH:.4e}, z_L diff={diff_zL:.4e}, steps diff={diff_steps}, halted diff={diff_halted}")

        if diff_loss > atol or diff_zH > atol or diff_zL > atol:
            print("DIVERGENCE above tolerance detected, stopping.")
            break

        if all_finish_pt and all_finish_jax:
            print("Both models signalled finish. Comparison done.")
            break

        if all_finish_pt != all_finish_jax:
            print("Halting flags diverged; stopping.")
            break

        step += 1
        if step > 50:
            print("Stopping after 50 steps to avoid infinite loop.")
            break


def main():
    parser = argparse.ArgumentParser(description="Step-by-step Torch vs JAX divergence checker")
    parser.add_argument("--ckpt", default="eval_data/sudoku/checkpoint.pth")
    parser.add_argument("--config", default="eval_data/sudoku/all_config.yaml")
    parser.add_argument("--atol", type=float, default=1)
    args = parser.parse_args()

    arch_cfg = load_arch_config(args.config)
    state_dict = torch_load_state_dict(args.ckpt)
    state_dict = {k[len("_orig_mod.model."):]: v for k, v in state_dict.items()}
    
    cfg = build_config_from_state_dict(arch_cfg, state_dict, name=args.ckpt)

    # Build core models
    pt_core = HRM_PT(cfg).eval()
    pt_core.load_state_dict(state_dict)

    key = jax.random.PRNGKey(0)
    jax_core = HRM_JAX(HRM_JAX_Config(**cfg), key=key)
    jax_core = convert_to_jax(jax_core, state_dict)

    # Wrap with ACTLossHead for consistent interface
    loss_name = arch_cfg.get("loss", {}).get("loss_type", "stablemax_cross_entropy") if isinstance(arch_cfg.get("loss"), dict) else arch_cfg.get("loss_type", "stablemax_cross_entropy")
    pt_model = ACTLossHead_PT(pt_core, loss_type=loss_name)

    jax_loss_fn = getattr(jax_losses, loss_name)
    jax_model = jax_losses.ACTLossHead(jax_core, loss_fn=jax_loss_fn)

    # ------------------------------------------------------------------
    # Load evaluation data (first batch of test split)
    # ------------------------------------------------------------------
    with open(args.config, "rt") as f:
        yaml_cfg = yaml.safe_load(f)

    full_cfg = PretrainConfig(**yaml_cfg)  # type: ignore

    eval_loader, _ = create_dataloader(full_cfg, split="test", rank=0, world_size=1, test_set_mode=True, epochs_per_iter=1, global_batch_size=full_cfg.global_batch_size)

    # Grab the very first batch
    set_name, batch_pt, _ = next(iter(eval_loader))

    print(f"Loaded batch from evaluation set '{set_name}' with shape {batch_pt['inputs'].shape}.")

    # Run step-by-step comparison
    step_compare(pt_model, jax_model, batch_pt, atol=args.atol)


if __name__ == "__main__":
    main() 