"""Shared diffusion schedule and repeatable Gaussian noise fields."""

import jax
import jax.numpy as jnp

STEPS = 200
_time = jnp.linspace(0, 1, STEPS + 1)
_curve = jnp.cos((_time + 0.008) / 1.008 * jnp.pi / 2) ** 2
BETAS = jnp.clip(1 - _curve[1:] / _curve[:-1], 0, 0.999)
ALPHA_BARS = jnp.cumprod(1 - BETAS)


def add(images: jax.Array, epsilon: jax.Array, timesteps: jax.Array) -> jax.Array:
    """Noise clean [-1, 1] images at the supplied timesteps."""
    alpha = ALPHA_BARS[timesteps][:, None, None, None]  # (B, 1, 1, 1)
    return jnp.sqrt(alpha) * images + jnp.sqrt(1 - alpha) * epsilon


def sample(key: jax.Array, images: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample independent timesteps and Gaussian noise for a batch."""
    time_key, noise_key = jax.random.split(key)
    timesteps = jax.random.randint(time_key, (images.shape[0],), 0, STEPS)
    epsilon = jax.random.normal(noise_key, images.shape)
    return add(images, epsilon, timesteps), epsilon, timesteps


def gaussian(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Draw Gaussian noise using the existing demo's seeded stream."""
    # Keep saved seeds stable by retaining the first stream of a three-stream draw.
    return jax.random.normal(key, (3, *shape))[0]


def velocity(images: jax.Array, epsilon: jax.Array, timesteps: jax.Array) -> jax.Array:
    """Stable diffusion target v = sqrt(alpha) * noise - sqrt(1-alpha) * image."""
    alpha = ALPHA_BARS[timesteps][:, None, None, None]
    return jnp.sqrt(alpha) * epsilon - jnp.sqrt(1 - alpha) * images
