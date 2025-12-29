# DPSN-JAX

Dynamic Parameter Selection Network implemented in JAX/Flax.

## Overview
This project implements the DPSN architecture which uses a router to dynamically select a sparse subset of parameters from a large pool for each token.

## Dependencies
- JAX (CPU)
- Flax
- Optax
- Grain

## Usage

### Run Tests
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 tests/test_dpsn.py
```

### Run Demo Training
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 main.py --steps 50
```

## Structure
- `src/dpsn/router.py`: Router logic (Complexity & Index selection).
- `src/dpsn/dpsn_layer.py`: The sparse parameter layer.
- `src/dpsn/model.py`: Full model architecture.
- `src/dpsn/data.py`: Data loading with Grain.
