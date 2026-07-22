# Installation

## Requirements

`pardiso-mkl-jax` targets Linux. Its oneMKL dependency is provided by
Intel's PyPI wheels, so no separate oneAPI or system MKL installation is
needed.

The package works with float64 values throughout, so your program needs
JAX's 64-bit mode enabled:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Set this once, near the start of your program, before creating any arrays
you plan to pass in. Without it, JAX silently truncates float64 arrays to
float32, and `pardiso_mkl_jax.solve` raises a clear `TypeError` rather than
solving the wrong system.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv add pardiso-mkl-jax
```

Or with pip:

```bash
pip install pardiso-mkl-jax
```

Both pull in `mkl`, the PyPI package providing the oneMKL shared libraries
Pardiso needs at runtime, alongside `jax`, `numpy`, and `scipy`.

## Building from source

Building the compiled extension requires a C++17 compiler and Cython, both
declared as build dependencies, so a plain `pip install` or `uv add` from
source handles this automatically. To work on the package itself:

```bash
git clone https://github.com/nardi/pardiso-mkl-jax
cd pardiso-mkl-jax
uv sync
uv run pytest
```
