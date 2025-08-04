from typing import Any, Tuple, Dict, Sequence, Optional, Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return jnp.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, axis=-1):
    s_x = s(x)
    return jnp.log(s_x / jnp.sum(s_x, axis=axis, keepdims=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.astype(jnp.float64), axis=-1)
    valid_mask = labels != ignore_index
    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(
        logprobs, transformed_labels[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    labels_one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, labels_one_hot)
    # mask ignore index
    loss = loss * (labels != ignore_index)
    return loss


class ACTLossHead(eqx.Module):
    model: eqx.Module
    loss_fn: Callable = eqx.field(static=True)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def __call__(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, jnp.ndarray, Dict[str, jnp.ndarray], Optional[Dict[str, jnp.ndarray]], jnp.ndarray]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = jnp.maximum(loss_counts, 1)[..., None]

        is_correct = mask & (jnp.argmax(outputs["logits"], axis=-1) == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts

        # Metrics (halted)
        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": valid_metrics.sum(),
            "accuracy": jnp.where(
                valid_metrics, (is_correct.astype(jnp.float32) / loss_divisor).sum(-1), 0
            ).sum(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
            "q_halt_accuracy": (
                valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
            ).sum(),
            "steps": jnp.where(valid_metrics, new_carry.steps, 0).sum(),
        }

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = optax.sigmoid_binary_cross_entropy(
            outputs["q_halt_logits"], seq_is_correct.astype(outputs["q_halt_logits"].dtype)
        ).sum()

        metrics.update({
            "lm_loss": lm_loss,
            "q_halt_loss": q_halt_loss,
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = optax.sigmoid_binary_cross_entropy(
                outputs["q_continue_logits"], outputs["target_q_continue"]
            ).sum()
            metrics["q_continue_loss"] = q_continue_loss

        # Filter outputs for return
        detached_outputs = {k: outputs[k] for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            jnp.all(new_carry.halted),
        )
