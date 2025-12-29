import yaml
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int
    num_layers: int
    num_memory_slots: int
    min_k: int
    max_k: int
    router_dim: int = 0  # Add router_dim field with default 0


@dataclass
class TrainingConfig:
    batch_size: int
    seq_len: int
    learning_rate: float
    steps: int
    log_every_steps: int
    save_every_steps: int
    seed: int
    workdir: str
 
    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)


@dataclass
class DataConfig:
    path: str
    vocab_path: str


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**cfg_dict["model"]),
            training=TrainingConfig(**cfg_dict["training"]),
            data=DataConfig(**cfg_dict["data"]),
        )


def load_config(path: str) -> Config:
    return Config.from_yaml(path)
