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
from loguru import logger

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.hrm_jax.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1Carry, HierarchicalReasoningModel_ACTV1InnerCarry
from models.hrm_jax.losses import ACTLossHead

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

from pretrain_jax import PretrainConfig, create_dataloader, init_train_state, evaluate

@hydra.main(config_path="eval_data/maze", config_name="all_config.yaml", version_base=None)
def launch(hydra_config: DictConfig):
    rank = jax.process_index()
    world_size = jax.local_device_count()

    # Create a mesh
    devices = mesh_utils.create_device_mesh((world_size, 1))
    mesh = Mesh(devices, axis_names=('data', 'model'))

    config = PretrainConfig(**hydra_config)


    # Dataset
    logger.info(f"Loading dataset")
    eval_loader,  eval_metadata  = create_dataloader(config, "test", rank=rank, world_size=world_size, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    try:
        n_samples = len(eval_loader.dataset)
    except:
        eval_loader.dataset._lazy_load_dataset()
        n_samples = len(next(iter(eval_loader.dataset._data.values()))["group_indices"])
    logger.info(f"Evaluating on {n_samples} examples")

    with mesh:
        # Train state
        logger.info(f"Initializing train state")
        train_state = init_train_state(config, eval_metadata, mesh)
        if config.checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {config.checkpoint_path}.")
            if os.path.exists(config.checkpoint_path):
                if config.checkpoint_path.endswith(".eqx"):
                    train_state = eqx.tree_deserialise_leaves(config.checkpoint_path, train_state)
                elif config.checkpoint_path.endswith(".pth"):
                    from pretrain_jax import load_pytorch_checkpoint
                    train_state = load_pytorch_checkpoint(config, train_state)
                else:
                    # Assuming it's a directory with a checkpoint
                    checkpoints = [f for f in os.listdir(config.checkpoint_path) if f.endswith(".eqx")]
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        train_state = eqx.tree_deserialise_leaves(os.path.join(config.checkpoint_path, latest_checkpoint), train_state)
                    else:
                        logger.error(f"No checkpoints found in {config.checkpoint_path}")
            else:
                logger.error(f"Checkpoint path {config.checkpoint_path} does not exist")
        else:
            logger.info(f"Not loading checkpoint.")

        logger.info(f"Evaluating")
        metrics = evaluate(train_state, eval_loader, total_batches=n_samples // config.global_batch_size)

    logger.info(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    launch()
