import argparse
import os
import jax

# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

from dpsn.inference import generate


def main():
    parser = argparse.ArgumentParser(description="Generate text with DPSN")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint pickle"
    )
    parser.add_argument("--prompt", type=str, default="The ", help="Prompt text")
    parser.add_argument("--length", type=int, default=200, help="Generation length")
    args = parser.parse_args()

    print(f"Running on {jax.devices()[0].platform.upper()}")
    generate(args.config, args.checkpoint, args.prompt, args.length)


if __name__ == "__main__":
    main()
