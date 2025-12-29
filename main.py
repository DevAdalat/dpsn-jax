import os
import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from dpsn.model import DPSNModel
from dpsn.data import create_dataset


def main():
    parser = argparse.ArgumentParser(description="Train DPSN on CPU")
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    args = parser.parse_args()

    print(f"Running DPSN on {jax.devices()[0].platform.upper()}")

    # Config
    D_MODEL = 64
    NUM_LAYERS = 2
    NUM_MEMORY = 1000
    MIN_K = 10
    MAX_K = 50

    # Data
    print("Creating dataset...")
    ds, vocab_size, _, _ = create_dataset(
        "data/input.txt", "data/vocab.json", args.batch_size, args.seq_len
    )
    iterator = iter(ds)

    # Model
    print("Initializing model...")
    model = DPSNModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_memory_slots=NUM_MEMORY,
        min_k=MIN_K,
        max_k=MAX_K,
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    init_rngs = {"params": init_rng, "noise": jax.random.PRNGKey(0)}

    dummy_input = jnp.ones((args.batch_size, args.seq_len), dtype=jnp.int32)
    params = model.init(init_rngs, dummy_input, training=False)["params"]

    # Optimizer
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Train Loop
    @jax.jit
    def train_step(state, batch, rng):
        def loss_fn(params):
            logits, (aux_loss, active_count) = state.apply_fn(
                {"params": params}, batch["input"], training=True, rngs={"noise": rng}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["target"]
            ).mean()

            # Add auxiliary loss with coefficient 0.01
            total_loss = loss + 0.01 * aux_loss
            metrics = {
                "loss": total_loss,
                "aux_loss": aux_loss,
                "active_count": active_count,
            }
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    print(f"Starting training for {args.steps} steps...")
    rng, step_rng = jax.random.split(rng)

    for i in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(ds)
            batch = next(iterator)

        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step(state, batch, step_rng)
        loss = metrics["loss"]
        active = metrics["active_count"]

        if i % 1 == 0:
            print(f"Step {i + 1}: Loss = {loss:.4f}, Active Params = {active:.0f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
