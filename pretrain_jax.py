from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
from models.hrm_jax import losses
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.hrm_jax.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1Carry, HierarchicalReasoningModel_ACTV1InnerCarry
from models.hrm_jax.losses import ACTLossHead

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


class TrainState(eqx.Module):
    model: eqx.Module
    tx: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState
    carry: Any
    key: jax.random.PRNGKey

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = HierarchicalReasoningModel_ACTV1Config(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    key = jax.random.PRNGKey(config.seed)
    model = HierarchicalReasoningModel_ACTV1(model_cfg, key=key)
    loss_function_name = getattr(config.arch.loss, 'loss_type', 'stablemax_cross_entropy')
    loss_fn = getattr(losses, loss_function_name)
    model = ACTLossHead(model, loss_fn=loss_fn)

    # Optimizers and lr
    tx = optax.adamw(
        learning_rate=config.lr,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay,
    )

    return model, tx, key


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, mesh: Mesh):
    world_size = mesh.size
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, tx, key = create_model(config, train_metadata, world_size=world_size)
    
    # Initialize with a dummy batch
    key, subkey = jax.random.split(key)
    batch_size = config.global_batch_size // world_size
    dummy_batch = {
        "inputs": jnp.ones((batch_size, model.model.config.seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, model.model.config.seq_len), dtype=jnp.int32),
        "puzzle_identifiers": jnp.ones((batch_size,), dtype=jnp.int32),
    }
    
    puzzle_emb_len = -(model.model.config.puzzle_emb_ndim // -model.model.config.hidden_size)
    inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
        z_H=jnp.zeros((batch_size, model.model.config.seq_len + puzzle_emb_len, model.model.config.hidden_size)),
        z_L=jnp.zeros((batch_size, model.model.config.seq_len + puzzle_emb_len, model.model.config.hidden_size)),
    )
    
    dummy_carry = HierarchicalReasoningModel_ACTV1Carry(
        inner_carry=inner_carry,
        steps=jnp.zeros((batch_size,), dtype=jnp.int32),
        halted=jnp.ones((batch_size,), dtype=jnp.bool_),
        current_data={k: jnp.zeros_like(v) for k, v in dummy_batch.items()},
    )

    def init_fn(model, carry):
        return model, carry

    def get_sharding(tree: Any):
        return jax.tree.map(
            lambda x: NamedSharding(mesh, PartitionSpec()),
            tree
        )

    model_sharding = get_sharding(eqx.filter(model, eqx.is_array))
    carry_sharding = get_sharding(dummy_carry)

    model = eqx.filter_shard(model, model_sharding)
    dummy_carry = eqx.filter_shard(dummy_carry, carry_sharding)

    opt_state = tx.init(eqx.filter(model, eqx.is_array))

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        tx=tx,
        opt_state=opt_state,
        carry=dummy_carry,
        key=key,
    )

def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(config.checkpoint_path, f"step_{train_state.step}.eqx"), train_state.model)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

@eqx.filter_jit(donate="all")
def train_batch(train_state: TrainState, batch: Any):
    if train_state.step >= train_state.total_steps:
        return train_state, {}

    key, new_key = jax.random.split(train_state.key)

    def loss_fn(model, carry, batch):
        new_carry, loss, metrics, _, _ = model(
            return_keys=[], carry=carry, batch=batch, key=key
        )
        return loss, (new_carry, metrics)

    (loss, (new_carry, metrics)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(train_state.model, train_state.carry, batch)
    
    updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.model)
    new_model = eqx.apply_updates(train_state.model, updates)

    metrics["train/loss"] = loss

    new_step = train_state.step + 1
    return TrainState(
        model=new_model,
        tx=train_state.tx,
        opt_state=new_opt_state,
        carry=new_carry,
        key=new_key,
        step=new_step,
        total_steps=train_state.total_steps,
    ), metrics

@eqx.filter_jit
def evaluate(train_state: TrainState, eval_loader: DataLoader):
    # TODO: Implement evaluation
    return {}

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-jax"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        return config
    else:
        # In a real multi-host setup, you'd broadcast the config
        return PretrainConfig(**hydra_config)


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    rank = jax.process_index()
    world_size = jax.local_device_count()

    # Create a mesh
    devices = mesh_utils.create_device_mesh((world_size, 1))
    mesh = Mesh(devices, axis_names=('data', 'model'))

    config = load_synced_config(hydra_config, rank, world_size)

    # Seed RNGs
    key = jax.random.PRNGKey(config.seed)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", rank=rank, world_size=world_size, test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", rank=rank, world_size=world_size, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)

    # Train state
    with mesh:
        train_state = init_train_state(config, train_metadata, mesh)

    # Progress bar and logger
    progress_bar = tqdm.tqdm(total=train_state.total_steps)

    wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
    # wandb.log({"num_params": sum(x.numel() for x in eqx.filter(train_state.model, eqx.is_array))}, step=0)
    save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        for _, batch, _ in train_loader:
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
                
            with mesh:    
                # Shard the batch to all devices
                batch_sharding = jax.tree.map(lambda x: NamedSharding(mesh, PartitionSpec('data')), batch)
                batch = jax.device_put(batch, batch_sharding)

                train_state, metrics = train_batch(train_state, batch)

            if metrics:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        ############ Evaluation
        metrics = evaluate(train_state, eval_loader)

        if metrics:
            wandb.log(metrics, step=train_state.step)
            
        ############ Checkpointing
        if config.checkpoint_every_eval or (_iter_id == total_iters - 1):
            save_train_state(config, train_state)

    wandb.finish()


if __name__ == "__main__":
    launch()

