import grain.python as grain
import numpy as np
import jax.numpy as jnp
import os
import json
from typing import Tuple, Dict, Any


class TextFileSource:
    def __init__(self, path: str, vocab_path: str, seq_len: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()

        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                vocab_data = json.load(f)
                self.chars = vocab_data["chars"]
                self.stoi = vocab_data["stoi"]
                self.itos = {int(k): v for k, v in vocab_data["itos"].items()}
        else:
            self.chars = sorted(list(set(self.text)))
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}

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


def create_dataset(
    path: str, vocab_path: str, batch_size: int, seq_len: int
) -> Tuple[grain.MapDataset, int, Dict, Dict]:
    source = TextFileSource(path, vocab_path, seq_len)

    indices = range(len(source))

    ds = grain.MapDataset.source(indices)
    ds = ds.map(source.__getitem__)
    ds = ds.shuffle(seed=42)
    ds = ds.batch(batch_size)

    return ds, source.vocab_size, source.stoi, source.itos
