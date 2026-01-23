import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import time
import os
from dpsn_jax.dpsn_flax import DPSNR, DPSNRConfig


# --- Data Loading ---
class TinyShakespeare:
    def __init__(self, path="data/input.txt", block_size=128):
        with open(path, "r") as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.data = np.array([self.stoi[c] for c in self.text], dtype=np.int32)
        self.block_size = block_size

        # Split 90/10
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split, batch_size, rng_key):
        data = self.train_data if split == "train" else self.val_data
        ix = jax.random.randint(rng_key, (batch_size,), 0, len(data) - self.block_size)

        x = np.stack([data[i : i + self.block_size] for i in ix])
        y = np.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        return jnp.array(x), jnp.array(y)

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])

    def encode(self, s):
        return [self.stoi[c] for c in s]


# --- Model & Train State ---
def create_train_state(model, params, learning_rate=3e-4):
    # AdamW with cosine decay (simplified to constant for short run)
    tx = optax.adamw(learning_rate, weight_decay=1e-2)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# --- Training Step ---
@jax.jit
def train_step(state, batch_x, batch_y, rng):
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        outputs = state.apply_fn(
            params, batch_x, train=True, rngs={"dropout": dropout_rng}
        )
        logits = outputs["logits"]

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch_y
        ).mean()
        ponder_loss = outputs["ponder_loss"]

        # Weighted sum
        loss = ce_loss + 0.01 * ponder_loss
        return loss, (ce_loss, ponder_loss, outputs["loops"])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, ponder_loss, loops)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, {
        "loss": loss,
        "ce_loss": ce_loss,
        "ponder_loss": ponder_loss,
        "loops": loops,
    }


# --- Generation ---
def generate(model, params, idx, max_new_tokens, dataset, rng):
    """Simple autoregressive generation."""

    # JIT the step function to avoid recompilation
    @jax.jit
    def next_token(params, idx_cond, rng):
        outputs = model.apply(params, idx_cond, train=False)
        logits = outputs["logits"][:, -1, :]
        rng, key = jax.random.split(rng)
        idx_next = jax.random.categorical(key, logits)
        return idx_next, key

    # We need to pad idx to max_seq_len for JIT
    max_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # Prepare fixed-size context
        # Take last max_len tokens
        curr_len = idx.shape[1]
        if curr_len < max_len:
            # Pad left with zeros
            pad_len = max_len - curr_len
            idx_cond = jnp.concatenate(
                [jnp.zeros((1, pad_len), dtype=jnp.int32), idx], axis=1
            )
        else:
            idx_cond = idx[:, -max_len:]

        # Forward
        idx_next, rng = next_token(params, idx_cond, rng)
        idx_next = idx_next[None]  # (1, 1)

        idx = jnp.concatenate([idx, idx_next], axis=1)

    return dataset.decode(idx[0])


# --- Main ---
def main():
    print("Loading data...")
    dataset = TinyShakespeare(block_size=64)  # Smaller context for speed
    print(f"Vocab size: {dataset.vocab_size}")

    # Config
    config = DPSNRConfig.nano()
    config.vocab_size = dataset.vocab_size
    config.max_seq_len = 64
    config.max_loops = 4
    config.dropout_rate = 0.1

    print(f"Initializing model (Params: {config.total_params})...")
    model = DPSNR(config)
    key = jax.random.PRNGKey(1337)

    # Init params
    key, init_key = jax.random.split(key)
    dummy_input = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(init_key, dummy_input)

    state = create_train_state(model, params)

    # Train Loop
    max_steps = 3000
    batch_size = 32
    eval_interval = 50

    print("Starting training...")
    start_time = time.time()

    for step in range(max_steps + 1):
        key, batch_key = jax.random.split(key)
        bx, by = dataset.get_batch("train", batch_size, batch_key)

        state, metrics = train_step(state, bx, by, key)

        if step % eval_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"\nStep {step} | t={elapsed:.1f}s | "
                f"Loss: {metrics['loss']:.4f} (CE: {metrics['ce_loss']:.4f}) | "
                f"Loops: {metrics['loops']:.2f}"
            )

            # Generate sample
            context = jnp.array(dataset.encode("\n"), dtype=jnp.int32)[None, :]
            gen_text = generate(model, state.params, context, 100, dataset, key)
            print("--- Generated Sample ---")
            print(gen_text)
            print("------------------------")


if __name__ == "__main__":
    main()
