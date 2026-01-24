import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax.training import train_state, orbax_utils
from datasets import load_dataset
from transformers import AutoTokenizer

# Sharding imports
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import jax_utils

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

import subprocess
import shutil

from dpsn_jax.dpsn_flax import DPSNR, DPSNRConfig


# --- Utils ---
def get_available_memory_mb():
    """Detects available device memory in MB."""
    platform = jax.local_devices()[0].platform

    if platform == "gpu":
        # Try nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,nounits,noheader",
                    ],
                    encoding="utf-8",
                    stdout=subprocess.PIPE,
                    check=False,
                )
                if result.returncode == 0:
                    # Return memory of first GPU
                    return int(result.stdout.strip().split("\n")[0])
            except Exception:
                pass
        # Fallback for GPU if nvidia-smi fails (safe default 8GB)
        return 8192

    elif platform == "tpu":
        # TPU memory detection is complex from Python.
        # TPU v2: 8GB, v3: 16GB, v4: 32GB, v5e: 16/32GB
        # We'll assume a conservative default (TPU v3)
        return 16384

    else:
        # CPU
        return 4096


def calculate_auto_batch_size(config, total_params, available_mem_mb):
    """
    Heuristic to calculate max batch size.
    """
    # 1. Static Memory
    # Convert params to MB (1 MB = 1e6 bytes approx for simple math)
    # Sharding Factor: With FSDP/Sharding, static memory per device is divided by num_devices
    num_devices = jax.device_count()

    # Each device holds 1/N params + optimizer state
    static_mem_bytes = (total_params * 18) / num_devices
    static_mem_mb = static_mem_bytes / (1024 * 1024)

    # 2. Available for Activations (Leave 10% buffer)
    usable_mem_mb = (available_mem_mb * 0.9) - static_mem_mb

    if usable_mem_mb <= 0:
        logging.warning("Model is too large for detected memory! Defaulting to BS=1")
        return 1

    # 3. Activation Cost per Sample (Heuristic)
    # FSDP activation sharding helps, but usually activations are sharded on batch dim.
    # So per device, we handle (Batch / NumDevices) samples.
    # The 'activation_bytes_per_sample' calculation below is for ONE sample.

    activation_bytes_per_sample = (
        config.max_seq_len * config.hidden_dim * config.num_layers * 24
    )
    # Add MassivePool overhead (Seq * PoolDim * MaxK * 4)
    activation_bytes_per_sample += (
        config.max_seq_len * config.pool_dim * config.max_k * 4 * 10
    )

    act_mem_mb_per_sample = activation_bytes_per_sample / (1024 * 1024)

    # Total batch capacity of the ENTIRE POD
    # Each device has 'usable_mem_mb'.
    # It can fit K samples.
    # Total samples = K * num_devices.

    samples_per_device = int(usable_mem_mb / act_mem_mb_per_sample)
    optimal_bs = samples_per_device * num_devices

    # Clamp
    optimal_bs = max(1, min(optimal_bs, 2048))  # Cap at 2048

    # Round down to nearest power of 2
    if optimal_bs > 1:
        optimal_bs = 2 ** int(np.log2(optimal_bs))

    return optimal_bs


# --- Configuration ---
@dataclass
class TrainArgs:
    # Model Config
    model_preset: str = "nano"
    max_loops: int = 8

    # Data Config
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-v1"
    dataset_split: str = "train"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    eos_token: Optional[str] = None
    max_seq_len: int = 128

    # Training Config
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    max_checkpoints: int = 3
    output_dir: str = "checkpoints"
    seed: int = 42
    resume: bool = True


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train DPSN-R (Sharded)")

    parser.add_argument(
        "--model_preset", type=str, default="nano", help="Model size preset"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext", help="HF Dataset name"
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="HF Dataset config"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split"
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Column name for text"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="gpt2", help="HF Tokenizer name"
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        help="Custom EOS token to append (e.g. <|endoftext|>)",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--max_loops", type=int, default=8, help="Maximum number of reasoning loops"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument(
        "--save_interval", type=int, default=500, help="Checkpoint interval"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--max_checkpoints", type=int, default=3, help="Max checkpoints to keep"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Auto resume from checkpoint",
    )

    args = parser.parse_args()
    return TrainArgs(**vars(args))


# --- Data Pipeline ---
def create_data_iterator(args: TrainArgs, tokenizer):
    """Creates a generator that yields batches of tokenized data."""
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    )

    iterator = iter(dataset)

    while True:
        batch_texts = []
        for _ in range(args.batch_size):
            try:
                item = next(iterator)
                text = item[args.text_column]
                if len(text.strip()) > 0:
                    if args.eos_token:
                        text = text + args.eos_token
                    batch_texts.append(text)
            except StopIteration:
                iterator = iter(dataset)
                continue

        if not batch_texts:
            continue

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
            return_tensors="np",
        )

        yield jnp.array(encodings["input_ids"])


# --- Generation ---
def generate(model, params, idx, max_new_tokens, tokenizer, rng, eos_token_id=None):
    # Generation is typically fast enough on single device or replicated.
    # For simplicity, we assume 'params' passed here are fully replicated or available on device.
    # In sharded training, params are sharded. We might need to unshard to host or replicate to generate.

    @jax.jit
    def next_token(params, idx_cond, rng):
        outputs = model.apply(params, idx_cond, train=False)
        logits = outputs["logits"][:, -1, :]
        rng, key = jax.random.split(rng)
        idx_next = jax.random.categorical(key, logits)
        return idx_next, key

    max_len = model.config.max_seq_len
    idx = jnp.array(idx)

    for _ in range(max_new_tokens):
        curr_len = idx.shape[1]
        if curr_len < max_len:
            pad_len = max_len - curr_len
            idx_cond = jnp.concatenate(
                [jnp.zeros((1, pad_len), dtype=jnp.int32), idx], axis=1
            )
        else:
            idx_cond = idx[:, -max_len:]

        idx_next, rng = next_token(params, idx_cond, rng)

        token_id = int(idx_next[0])
        if eos_token_id is not None and token_id == eos_token_id:
            break

        idx_next = idx_next[None]
        idx = jnp.concatenate([idx, idx_next], axis=1)

    return tokenizer.decode(np.array(idx[0]), skip_special_tokens=True)


# --- Training Logic ---
def create_train_state(model, params, args):
    warmup_steps = int(0.1 * args.num_steps)
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate * 0.1,
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=args.num_steps,
        end_value=args.learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler, weight_decay=0.01),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def main():
    print("Starting SHARDED training script...")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    # 1. Setup Mesh
    try:
        devices = jax.devices()
        num_devices = len(devices)
        print(f"JAX Devices: {num_devices} ({devices[0].platform})")
        logging.info(f"JAX Devices: {devices}")

        # Create 1D mesh for FSDP
        # Axis 'fsdp' will be used to shard parameters
        # Axis 'batch' will be used to shard data (but here we map both to same mesh axis usually?)
        # Standard FSDP: 1D mesh.
        # Data batch is sharded on this axis.
        # Params are sharded on this axis (axis 0 of kernels).

        mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=("fsdp",))
        logging.info(f"Mesh created: {mesh}")

    except Exception as e:
        print(f"Error checking devices: {e}")
        return

    args = parse_args()
    print(f"Configuration parsed: {args.model_preset}")
    logging.info(f"Configuration: {args}")

    # 2. Tokenizer
    logging.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Config
    if args.model_preset == "nano":
        config = DPSNRConfig.nano()
    elif args.model_preset == "micro":
        config = DPSNRConfig.micro()
    elif args.model_preset == "mini":
        config = DPSNRConfig.mini()
    elif args.model_preset == "small":
        config = DPSNRConfig.small()
    elif args.model_preset == "small_300m":
        config = DPSNRConfig.small_300m()
    elif args.model_preset == "medium":
        config = DPSNRConfig.medium()
    elif args.model_preset == "large":
        config = DPSNRConfig.large()
    else:
        raise ValueError(f"Unknown preset: {args.model_preset}")

    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.max_seq_len
    config.max_loops = args.max_loops

    # 4. Initialize Model (Abstract)
    logging.info("Initializing model...")
    model = DPSNR(config)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    # We init on CPU/Abstract first to avoid OOM before sharding
    # Or use jit init with sharding.

    # Define Sharding Specs
    # We want to shard parameters on 'fsdp' axis.
    # Simple heuristic: Shard the largest dimension?
    # Or just shard the first dimension (axis 0) of every array?
    # Standard FSDP often shards axis 0.

    # P(None) means replicated. P('fsdp') means sharded.
    # We define a PyTree of PartitionSpecs matching the params structure.
    # But doing this manually is tedious.
    # We can use jax.sharding.PositionalSharding.

    # BETTER: Use a rule. "Shard axis 0 of all params".
    # But scalars (dim 0) cannot be sharded.
    # Flax params are usually kernels (FeaturesIn, FeaturesOut) or biases (FeaturesOut).

    dummy_input = jnp.zeros((1, args.max_seq_len), dtype=jnp.int32)

    # Abstract initialization to get shape
    abstract_variables = jax.eval_shape(lambda: model.init(init_key, dummy_input))
    abstract_params = abstract_variables["params"]

    # Define Sharding Rule
    def get_fsdp_sharding(x):
        # Shard first dimension if it is divisible by mesh size?
        # Or just allow uneven sharding?
        # FSDP usually shards dim 0.
        if x.ndim >= 1:
            return NamedSharding(mesh, P("fsdp"))
        return NamedSharding(mesh, P())  # Replicated scalars

    # Create PyTree of shardings
    param_shardings = jax.tree_util.tree_map(get_fsdp_sharding, abstract_params)

    # JIT Init with sharding
    @jax.jit
    def sharded_init(key, x):
        return model.init(key, x)

    # We need to tell JIT that output 'params' should be sharded according to 'param_shardings'
    # But model.init returns {'params': ..., 'dropout': ...}
    # We handle params only.

    # Actually, simpler: Initialize on host (if fits RAM), then jax.device_put with sharding.
    # If 1B model, it fits in host RAM (300GB).

    logging.info("Initializing params on host...")
    variables = model.init(init_key, dummy_input)
    params = variables["params"]

    # Move to Device with Sharding
    logging.info("Sharding params to device...")
    params = jax.device_put(params, param_shardings)

    # --- Auto Batch Size ---
    if args.batch_size <= 0:
        logging.info("Auto-tuning batch size...")
        avail_mem = get_available_memory_mb()
        # avail_mem is per device.
        args.batch_size = calculate_auto_batch_size(
            config, config.total_params, avail_mem
        )
        logging.info(f"Auto-selected Batch Size: {args.batch_size}")
        print(f"Auto-selected Batch Size: {args.batch_size}")

    logging.info(f"Global Batch Size: {args.batch_size}")

    # Create Train State (Sharded)
    # The optimizer state will follow the sharding of params automatically if we use standard optax?
    # Yes, usually.
    state = create_train_state(model, params, args)

    # We need to ensure state.opt_state is also sharded.
    # create_train_state calls init logic.
    # We should probably JIT create_train_state with sharding constraints or device_put the result.
    # But since 'params' is already sharded, optax might init state on device.
    # To be safe, let's re-put the whole state.

    state_sharding = jax.tree_util.tree_map(get_fsdp_sharding, state)
    state = jax.device_put(state, state_sharding)

    logging.info(f"Model Parameters: {config.total_params:,}")

    # 5. Checkpoint Manager
    abs_output_dir = os.path.abspath(args.output_dir)
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=args.max_checkpoints, create=True
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        abs_output_dir, orbax.checkpoint.PyTreeCheckpointer(), options=options
    )

    start_step = 0
    if args.resume:
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            logging.info(f"Resuming from step {latest_step}...")
            # Restore
            # Orbax can restore into sharded arrays if we provide target
            state = checkpoint_manager.restore(latest_step, items=state)
            start_step = latest_step + 1
        else:
            logging.info("No checkpoint found, starting from scratch.")

    writer = SummaryWriter(log_dir=os.path.join(abs_output_dir, "tb"))

    # 6. Train Step
    # Data Sharding: We want batch dim (0) to be sharded on 'fsdp' axis.
    data_sharding = NamedSharding(mesh, P("fsdp"))

    @jax.jit
    def train_step(state, batch, rng):
        dropout_rng = jax.random.fold_in(rng, state.step)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        def loss_fn(params):
            outputs = state.apply_fn(
                params, inputs, train=True, rngs={"dropout": dropout_rng}
            )
            logits = outputs["logits"]
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()
            ponder_loss = outputs["ponder_loss"]
            loss = ce_loss + 0.01 * ponder_loss
            return loss, (ce_loss, ponder_loss, outputs["loops"])

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)

        # In FSDP (Sharding), we don't need manual pmean usually if using pjit with correct PartitionSpec?
        # WAIT: 'grads' produced by value_and_grad on sharded params will be sharded.
        # But are they fully reduced?
        # With jax.jit and sharded inputs/weights, JAX automatically inserts collectives (all-reduce/reduce-scatter).
        # We generally DO NOT need explicit pmean inside jit-ted function when using sharding API.

        state = state.apply_gradients(grads=grads)
        return state, metrics

    # 7. Training Loop
    data_iter = create_data_iterator(args, tokenizer)

    logging.info(f"Starting training from step {start_step}...")
    start_time = time.time()

    # Save args
    # We need to unshard to save? Orbax handles distributed checkpoints.
    # But 'save_args' usually wants replicated or simple layout.
    # Let's keep it simple.

    avg_loss = 0.0
    steps_in_epoch = 0

    for step in range(start_step, args.num_steps):
        batch_cpu = next(data_iter)

        # Put batch onto device with sharding
        # This distributes the batch across the mesh
        with mesh:
            batch = jax.device_put(batch_cpu, data_sharding)

            # Step Key
            step_key = jax.random.fold_in(key, step)

            step_start = time.time()
            state, metrics = train_step(state, batch, step_key)
            step_time = time.time() - step_start

            # Metrics are likely sharded/replicated. Pull to host.
            # We can just take float() which triggers sync
            loss_val = float(metrics["loss"])

            avg_loss = (avg_loss * steps_in_epoch + loss_val) / (steps_in_epoch + 1)
            steps_in_epoch += 1

            tps = (args.batch_size * args.max_seq_len) / step_time

            writer.add_scalar("train/loss", loss_val, step)
            writer.add_scalar("train/ce_loss", float(metrics["ce_loss"]), step)
            writer.add_scalar("train/ponder_loss", float(metrics["ponder_loss"]), step)
            writer.add_scalar("train/loops", float(metrics["loops"]), step)
            writer.add_scalar("perf/tps", tps, step)
            writer.add_scalar("perf/step_time", step_time, step)

            if step % args.eval_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step - start_step + 1) / elapsed if elapsed > 0 else 0

                logging.info(
                    f"Step {step}/{args.num_steps} | "
                    f"Loss: {loss_val:.4f} (Avg: {avg_loss:.4f}) | "
                    f"Loops: {float(metrics['loops']):.2f} | "
                    f"TPS: {tps:.0f} | "
                    f"Steps/s: {steps_per_sec:.2f} | "
                    f"Elapsed: {elapsed / 60:.1f}m"
                )

                try:
                    # Generation: We need to unshard params to host or replicate?
                    # Or run generation on device.
                    # Simplest: Unshard params to host, then run generate (jit-ted) on CPU or single device.
                    # Or just pass sharded params to generate?
                    # If generate uses jit, it should handle sharded inputs.
                    # But generation is autoregressive (loop).

                    # Let's try passing sharded params directly.
                    # But we need inputs to be on device.

                    prompt = "Once upon a time"
                    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
                    input_ids = jnp.array(input_ids)

                    # Replicate input to mesh? Or just put on device 0?
                    # Generate usually runs on batch=1.
                    # Sharding batch=1 across 8 devices is weird.
                    # Let's fully replicate params for generation (temporary).
                    # Fully replicated params might OOM if model is 1B+?
                    # 1B float32 = 4GB. Replicated fits on one chip.
                    # So we can "respec" params to P(None) (Replicated) temporarily?

                    # For now, let's try running generate with sharded params.
                    # Input (1, Len). Sharding P('fsdp') means splitting batch 1? No.
                    # Input should be P(None) (Replicated).

                    gen_text = generate(
                        model,
                        state.params,  # Sharded
                        input_ids,
                        50,
                        tokenizer,
                        jax.random.PRNGKey(42),
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    print(f"--- Generated Sample (Step {step}) ---")
                    print(gen_text)
                    print("-------------------------------------")
                    writer.add_text("gen/sample", gen_text, step)
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                    import traceback

                    traceback.print_exc()

            if step > 0 and step % args.save_interval == 0:
                checkpoint_manager.save(step, state)
                logging.info(f"Saved checkpoint step {step}")

    checkpoint_manager.save(args.num_steps, state)
    writer.close()
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
