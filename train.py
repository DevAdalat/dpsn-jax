import argparse
import os
import jax

# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

from dpsn.config import load_config
from dpsn.train import train


def main():
    parser = argparse.ArgumentParser(description="Train DPSN")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    print(f"Running on {jax.devices()[0].platform.upper()}")
    train(config)


if __name__ == "__main__":
    main()
