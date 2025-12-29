import grain.python as grain
import numpy as np
import jax.numpy as jnp
import os
import json
from typing import Tuple, Dict, Any, Optional, Iterator, Union, Iterable, cast

# Initialize optional imports
hf_load_dataset = None
AutoTokenizer = None
HF_AVAILABLE = False

try:
    from datasets import load_dataset as hf_load_dataset  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    HF_AVAILABLE = True
except ImportError:
    pass


class TextFileSource:
    def __init__(self, path: str, vocab_path: Optional[str], seq_len: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()

        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                vocab_data = json.load(f)
                self.chars = vocab_data["chars"]
                self.stoi = vocab_data["stoi"]
                self.itos = {int(k): v for k, v in vocab_data["itos"].items()}
        else:
            self.chars = sorted(list(set(self.text)))
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}

            if vocab_path:
                os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                with open(vocab_path, "w") as f:
                    json.dump(
                        {"chars": self.chars, "stoi": self.stoi, "itos": self.itos}, f
                    )

        self.vocab_size = len(self.chars)
        self.data = np.array([self.stoi[c] for c in self.text], dtype=np.int32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return {"input": x, "target": y}


def hf_data_generator(dataset, tokenizer, seq_len, column_name):
    # Buffer to hold tokens
    buffer = []

    # Iterate through the dataset
    for example in dataset:
        text = example[column_name]
        # Skip empty text
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)

        while len(buffer) >= seq_len + 1:
            x = buffer[:seq_len]
            y = buffer[1 : seq_len + 1]
            yield {
                "input": np.array(x, dtype=np.int32),
                "target": np.array(y, dtype=np.int32),
            }
            buffer = buffer[seq_len:]


def create_dataset(
    path: Optional[str],
    vocab_path: Optional[str],
    batch_size: int,
    seq_len: int,
    huggingface_dataset: Optional[str] = None,
    huggingface_dataset_config: Optional[str] = None,
    dataset_column_name: str = "text",
    tokenizer_name: Optional[str] = None,
    streaming: bool = True,
) -> Tuple[Iterable, int, Any, Any]:
    if huggingface_dataset:
        if not HF_AVAILABLE:
            raise ImportError(
                "Please install datasets and transformers to use Hugging Face datasets."
            )

        if not tokenizer_name:
            raise ValueError(
                "tokenizer_name is required when using huggingface_dataset"
            )

        # Suppress type errors for optional imports
        # We checked HF_AVAILABLE so AutoTokenizer is not None here
        if AutoTokenizer is None or hf_load_dataset is None:
            raise ImportError("Failed to import datasets or transformers")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )  # type: ignore

        vocab_size = tokenizer.vocab_size  # type: ignore

        # Load dataset
        ds = hf_load_dataset(  # type: ignore
            huggingface_dataset,
            huggingface_dataset_config,
            split="train",
            streaming=streaming,
            trust_remote_code=True,
        )

        if streaming:
            # Create a generator that yields processed examples
            def gen():
                return hf_data_generator(ds, tokenizer, seq_len, dataset_column_name)

            class BatchIterator:
                def __init__(self, generator_func, batch_size):
                    self.gen_func = generator_func
                    self.batch_size = batch_size
                    self.iterator = self.gen_func()

                def __iter__(self):
                    return self

                def __next__(self):
                    inputs = []
                    targets = []
                    for _ in range(self.batch_size):
                        try:
                            item = next(self.iterator)
                        except StopIteration:
                            self.iterator = self.gen_func()  # Restart
                            item = next(self.iterator)
                        inputs.append(item["input"])
                        targets.append(item["target"])

                    return {"input": np.stack(inputs), "target": np.stack(targets)}

            dataset_iterator = BatchIterator(gen, batch_size)
            return dataset_iterator, vocab_size, None, None

    # Fallback to local file
    if path is None:
        raise ValueError("Either path or huggingface_dataset must be provided.")

    source = TextFileSource(path, vocab_path, seq_len)
    indices = range(len(source))

    ds = grain.MapDataset.source(indices)
    ds = ds.map(source.__getitem__)
    ds = ds.shuffle(seed=42)
    ds = ds.batch(batch_size)

    return ds, source.vocab_size, source.stoi, source.itos
