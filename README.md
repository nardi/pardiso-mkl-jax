# pardiso-mkl-jax

JAX-compatible interface to the oneMKL Pardiso direct sparse solver.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pmj

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

x = pmj.solve(
    indptr, indices, values, right_hand_side, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
)
```

Install with `uv add pardiso-mkl-jax` (or `pip install pardiso-mkl-jax`) and
[read the documentation here](https://nardi.github.io/pardiso-mkl-jax).

## Development

```bash
uv sync
uv run pytest
uv run ty check
```

The compiled extension rebuilds automatically on import while developing, so
the default editable install (`uv sync`) is what you want day to day. Building
the documentation site needs a non-editable install instead, since
mkdocstrings' static analysis cannot resolve import aliases through the
editable loader:

```bash
uv sync --group docs --no-editable
uv run --no-sync zensical build
uv run --no-sync pytest --markdown-docs docs
```
