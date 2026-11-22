"""One Flax MessagePack file per model."""

import pathlib

import flax.serialization
import jax
from flax import nnx


def save(model: nnx.Module, path: pathlib.Path) -> None:
    """Replace this model's final weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = jax.device_get(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(flax.serialization.to_bytes(state))
    temporary.replace(path)


def load(model: nnx.Module, path: pathlib.Path) -> None:
    """Restore parameters onto the current JAX device."""
    state = nnx.state(model, nnx.Param)
    restored = flax.serialization.from_bytes(nnx.to_pure_dict(state), path.read_bytes())
    nnx.replace_by_pure_dict(state, jax.device_put(restored))
    nnx.update(model, state)
