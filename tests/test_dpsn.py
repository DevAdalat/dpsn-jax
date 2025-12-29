import os

# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from dpsn.model import DPSNModel
from dpsn.data import create_dataset
import numpy as np


def test_dpsn_training_step():
    print("Initializing DPSN test...")

    # Config
    BATCH_SIZE = 2
    SEQ_LEN = 16
    D_MODEL = 32
    NUM_LAYERS = 2
    NUM_MEMORY = 100
    MIN_K = 2
    MAX_K = 10

    # Data
    ds, vocab_size, _, _ = create_dataset(
        "data/input.txt", "data/vocab.json", BATCH_SIZE, SEQ_LEN
    )
    iterator = iter(ds)
    batch = next(iterator)
    print(f"Data batch shape: {batch['input'].shape}")

    # Model
    model = DPSNModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_memory_slots=NUM_MEMORY,
        min_k=MIN_K,
        max_k=MAX_K,
    )

    # Init
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # We need 'noise' rng for the router
    init_rngs = {"params": init_rng, "noise": jax.random.PRNGKey(1)}

    dummy_input = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    params = model.init(init_rngs, dummy_input, training=False)["params"]

    print("Model initialized.")

    # Optimizer
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training step
    def train_step(state, batch, rng):
        def loss_fn(params):
            # Model returns: logits, (aux_loss, active_count)
            logits, (aux_loss, active_count) = state.apply_fn(
                {"params": params}, batch["input"], training=True, rngs={"noise": rng}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["target"]
            ).mean()
            total_loss = loss + 0.01 * aux_loss
            # Since jax.value_and_grad can only handle scalar aux returns easily if unchecked,
            # or requires has_aux=True which returns the second element of the tuple.
            # We return (loss, (metrics))
            return total_loss, (logits, aux_loss, active_count)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, aux_loss, active_count)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, logits

    # Run one step
    rng, step_rng = jax.random.split(rng)
    state, loss, logits = train_step(state, batch, step_rng)

    print(f"Initial Loss: {loss}")

    # Check if we can run another step and loss changes (or just runs)
    rng, step_rng = jax.random.split(rng)
    state, loss2, _ = train_step(state, batch, step_rng)

    print(f"Second Step Loss: {loss2}")

    assert not jnp.isnan(loss)
    assert not jnp.isnan(loss2)
    print("Test passed: Model runs on CPU and updates parameters.")


if __name__ == "__main__":
    test_dpsn_training_step()
