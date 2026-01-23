import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import linen as nn
import time
from typing import Tuple, Any

from .dpsn_flax import DPSNR, DPSNRConfig


def create_train_state(model, params, learning_rate=1e-3):
    """Creates initial TrainState."""
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=0.01),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def create_synthetic_data(key, batch_size, seq_len, vocab_size):
    """Generate random sequences for training."""
    return jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)


@jax.jit
def train_step(
    state: train_state.TrainState, batch: jnp.ndarray, rng: jax.random.PRNGKey
) -> Tuple[train_state.TrainState, Any]:
    """Performs a single training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        outputs = state.apply_fn(
            params, batch[:, :-1], train=True, rngs={"dropout": dropout_rng}
        )
        logits = outputs["logits"]
        targets = batch[:, 1:]

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()
        ponder_loss = outputs["ponder_loss"]

        # Ponder loss weight should come from config, but here we hardcode or pass it
        # Assuming typical weight
        total_loss = ce_loss + 0.01 * ponder_loss
        return total_loss, (ce_loss, ponder_loss, outputs["loops"])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, ponder_loss, loops)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    metrics = {
        "loss": loss,
        "ce_loss": ce_loss,
        "ponder_loss": ponder_loss,
        "loops": loops,
    }
    return state, metrics


def train_loop(
    config: DPSNRConfig, num_steps: int = 100, batch_size: int = 4, seed: int = 0
):
    """Runs a simple training loop."""
    key = jax.random.PRNGKey(seed)
    model = DPSNR(config)

    key, init_key, data_key = jax.random.split(key, 3)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(init_key, dummy_input)

    state = create_train_state(model, params)

    metrics_history = []

    print(f"Starting training for {num_steps} steps...")
    start_time = time.time()

    for step in range(num_steps):
        key, step_key = jax.random.split(key)
        batch = create_synthetic_data(
            step_key, batch_size, config.max_seq_len, config.vocab_size
        )

        state, metrics = train_step(state, batch, key)
        metrics_history.append(metrics)

        if step % 10 == 0:
            print(
                f"Step {step}: Loss={metrics['loss']:.4f}, "
                f"CE={metrics['ce_loss']:.4f}, "
                f"Ponder={metrics['ponder_loss']:.4f}, "
                f"Loops={metrics['loops']:.2f}"
            )

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")

    return metrics_history
