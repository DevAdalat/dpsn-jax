import jax
import jax.numpy as jnp
import optax
import time
import os
import resource
import pickle
import gc
from flax.training import train_state
from flax import linen as nn
from typing import Any, cast

from .model import DPSNModel
from .config import Config


class TrainState(train_state.TrainState):
    rngs: Any


def save_checkpoint(state, workdir, step):
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"checkpoint_{step}.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {"params": state.params, "step": step, "opt_state": state.opt_state}, f
        )
    print(f"Saved checkpoint to {path}")


def restore_checkpoint(path, model, dummy_input):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Restored checkpoint from {path}")
    return data


def train(config: Config):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    print("Initializing dataset...")
    from .data import create_dataset

    ds, vocab_size, stoi, itos = create_dataset(
        path=config.data.path,
        vocab_path=config.data.vocab_path,
        batch_size=config.training.batch_size,
        seq_len=config.training.seq_len,
        huggingface_dataset=config.data.huggingface_dataset,
        huggingface_dataset_config=config.data.huggingface_dataset_config,
        dataset_column_name=config.data.dataset_column_name,
        tokenizer_name=config.data.tokenizer_name,
        streaming=config.data.streaming,
    )
    iterator = iter(ds)

    print("Initializing model...")
    model = DPSNModel(
        vocab_size=vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_memory_slots=config.model.num_memory_slots,
        min_k=config.model.min_k,
        max_k=config.model.max_k,
        router_dim=config.model.router_dim,
    )

    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)

    dummy_input = jnp.ones(
        (config.training.batch_size, config.training.seq_len), dtype=jnp.int32
    )
    init_rngs = {"params": init_rng, "noise": jax.random.PRNGKey(0)}
    params = model.init(init_rngs, dummy_input, training=False)["params"]

    print("Calculating model statistics...")
    from flax.traverse_util import flatten_dict

    flat_params = flatten_dict(params)

    def get_size(v):
        return v.size

    total_params = sum(get_size(v) for v in flat_params.values())
    router_params = sum(get_size(v) for k, v in flat_params.items() if "router" in k)
    pool_params = sum(get_size(v) for k, v in flat_params.items() if "param_pool" in k)
    other_params = total_params - router_params - pool_params

    del flat_params
    gc.collect()

    print("\n" + "=" * 50)
    print(f"Model Architecture Statistics")
    print("=" * 50)
    print(f"Total Parameters:       {total_params:,}")
    print(
        f"  - Parameter Pool:     {pool_params:,} ({pool_params / total_params * 100:.1f}%)"
    )
    print(
        f"  - Router Networks:    {router_params:,} ({router_params / total_params * 100:.1f}%)"
    )
    print(
        f"  - Backbone (Attn/LN): {other_params:,} ({other_params / total_params * 100:.1f}%)"
    )
    print("-" * 50)
    print(f"Config:")
    print(f"  - Layers: {config.model.num_layers}")
    print(f"  - D_Model: {config.model.d_model}")
    print(f"  - Memory Slots: {config.model.num_memory_slots}")
    print("=" * 50 + "\n")

    print("Creating train state...")
    tx = optax.adam(learning_rate=config.training.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, rngs=init_rngs
    )

    del params
    gc.collect()

    @jax.jit(donate_argnums=(0,))
    def train_step(state, batch, rng):
        def loss_fn(params):
            logits, (aux_loss, active_count) = state.apply_fn(
                {"params": params}, batch["input"], training=True, rngs={"noise": rng}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["target"]
            ).mean()

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

    print(f"Starting training for {config.training.steps} steps...")

    for step in range(1, config.training.steps + 1):
        t0 = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            print("Iterator exhausted. Restarting...")
            iterator = iter(ds)
            batch = next(iterator)
        load_time = time.time() - t0

        batch_jax = {
            "input": jnp.array(batch["input"], dtype=jnp.int32),
            "target": jnp.array(batch["target"], dtype=jnp.int32),
        }

        rng, step_rng = jax.random.split(rng)

        if step == 1:
            print("Compiling train_step... (This may take a while and use memory)")
            mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"Memory before compilation: {mem_before:.2f} MB")

        t1 = time.time()
        state, metrics = train_step(state, batch_jax, step_rng)

        jax.block_until_ready(metrics["loss"])
        step_time = time.time() - t1

        loss = metrics["loss"]
        active = metrics["active_count"]

        if step % config.training.log_every_steps == 0 or step == 1:
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(
                f"Step {step}: Loss = {loss:.4f}, Active Params = {active:.0f}, "
                f"Load Time: {load_time:.4f}s, Step Time: {step_time:.4f}s, Memory: {mem_usage:.2f} MB"
            )

        if step % config.training.save_every_steps == 0:
            save_checkpoint(state, config.training.workdir, step)
            gc.collect()

    save_checkpoint(state, config.training.workdir, config.training.steps)
    print("Training complete.")
