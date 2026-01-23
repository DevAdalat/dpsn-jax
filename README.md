# DPSN-R JAX

JAX/Flax implementation of **Dynamic Parameter Selection Network with Recurrent Reasoning (DPSN-R)**.

## Features
- **TPU Optimized**: Static shapes, `jax.lax.scan` for recurrence, and `jax.sharding` for massive pool.
- **Efficient**: Decoupled Memory (~100B params) from Compute (~1B params active).
- **Hugging Face Compatible**: Train on any HF dataset with any HF tokenizer.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

Train a small model on TinyStories:

```bash
python src/dpsn_jax/train_full.py \
    --model_preset small \
    --tokenizer_name "EleutherAI/gpt-neo-125m" \
    --dataset_name "roneneldan/TinyStories" \
    --batch_size 16 \
    --num_steps 1000
```

## Config Presets

| Preset | Hidden | Layers | Pool Size | Params |
|--------|--------|--------|-----------|--------|
| nano   | 64     | 2      | 1k        | ~150k  |
| micro  | 128    | 2      | 10k       | ~3M    |
| small  | 768    | 12     | 400k      | ~400M  |
| medium | 1024   | 16     | 400k      | ~600M  |
| large  | 1280   | 24     | 500k      | ~1B    |
