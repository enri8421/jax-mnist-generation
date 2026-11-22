"""Two generation methods returning three progressively refined images."""

import enum

import jax
import jax.numpy as jnp
from flax import nnx

from models import diffusion, tokenizer, transformer
from utils import noise

REFINE_STEPS = 60
TOKEN_TEMPERATURE = 0.7
TOKEN_TOP_P = 0.95


class Method(enum.StrEnum):
    TOKENS = "tokens"
    DIFFUSION = "diffusion"


class Tokens(nnx.Module):
    def __init__(self):
        self.codec = tokenizer.Tokenizer(rngs=nnx.Rngs(0))
        self.model = transformer.Transformer(rngs=nnx.Rngs(1))

    def __call__(
        self,
        labels: jax.Array,
        key: jax.Array,
        refinement_key: jax.Array,
    ) -> jax.Array:
        context = jnp.zeros((labels.shape[0], 7, 7, 32))
        previews = []
        for level, size in enumerate(tokenizer.SCALES):
            stage_key = refinement_key if level == 2 else jax.random.fold_in(key, level)
            logits = self.model(context, labels, level)  # (B, size*size, 64)
            codes = _sample_codes(stage_key, logits).reshape(-1, size, size)
            context = context + self.codec.embed_level(codes, level)
            previews.append(self.codec.decoder(context))
        return jnp.stack(previews)  # (3, B, 28, 28, 1)


class Diffusion(nnx.Module):
    def __init__(self):
        self.model = diffusion.Denoiser(rngs=nnx.Rngs(2))

    def __call__(
        self,
        labels: jax.Array,
        key: jax.Array,
        refinement_key: jax.Array,
    ) -> jax.Array:
        initial = noise.gaussian(jax.random.fold_in(key, noise.STEPS), (labels.shape[0], 28, 28, 1))
        middle = _denoise(
            self.model,
            initial,
            labels,
            key,
            refinement_key,
            jnp.arange(noise.STEPS - 1, REFINE_STEPS - 1, -1),
        )
        final = _denoise(
            self.model,
            middle,
            labels,
            key,
            refinement_key,
            jnp.arange(REFINE_STEPS - 1, -1, -1),
        )
        return jnp.clip((jnp.stack((initial, middle, final)) + 1) / 2, 0, 1)


def sample(
    method: Method,
    models: dict[Method, Tokens | Diffusion],
    label: int,
    seed: int,
    refinement_seed: int,
) -> jax.Array:
    """Select a loaded method and return three images shaped (3, 28, 28, 1)."""
    return _sample(models[Method(method)], label, seed, refinement_seed)


@nnx.jit
def _sample(model: Tokens | Diffusion, label: int, seed: int, refinement_seed: int) -> jax.Array:
    labels = jnp.array([label], dtype=jnp.int32)
    return model(labels, jax.random.key(seed), jax.random.key(refinement_seed))[:, 0]


def _sample_codes(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Sample likely codes while retaining the token that crosses the nucleus threshold."""
    logits = logits / TOKEN_TEMPERATURE
    order = jnp.argsort(-logits, axis=-1)
    probabilities = jax.nn.softmax(jnp.take_along_axis(logits, order, axis=-1), axis=-1)
    keep = jnp.cumsum(probabilities, axis=-1) - probabilities < TOKEN_TOP_P
    keep = jnp.take_along_axis(keep, jnp.argsort(order, axis=-1), axis=-1)
    return jax.random.categorical(key, jnp.where(keep, logits, -jnp.inf))


def _denoise(
    model: diffusion.Denoiser,
    images: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    refinement_key: jax.Array,
    times: jax.Array,
) -> jax.Array:
    def update(images: jax.Array, pair: tuple[jax.Array, jax.Array]):
        current, following = pair
        timesteps = jnp.full(labels.shape, current, dtype=jnp.int32)
        # One batched forward pass for unconditional and digit-conditioned v.
        predictions = model(
            jnp.concatenate((images, images)),
            jnp.concatenate((timesteps, timesteps)),
            jnp.concatenate((jnp.full_like(labels, 10), labels)),
        )
        unconditional, conditional = jnp.split(predictions, 2)
        velocity = unconditional + 2.0 * (conditional - unconditional)
        alpha = noise.ALPHA_BARS[current]
        previous_alpha = jnp.where(following < 0, 1.0, noise.ALPHA_BARS[jnp.maximum(following, 0)])
        # v gives clean images without division by a near-zero signal coefficient.
        clean = jnp.clip(jnp.sqrt(alpha) * images - jnp.sqrt(1 - alpha) * velocity, -1, 1)
        step_alpha = alpha / previous_alpha
        beta = 1 - step_alpha
        variance = beta * (1 - previous_alpha) / (1 - alpha)
        mean = (
            jnp.sqrt(previous_alpha) * beta / (1 - alpha) * clean
            + jnp.sqrt(step_alpha) * (1 - previous_alpha) / (1 - alpha) * images
        )
        stage_key = jnp.where(current < REFINE_STEPS, refinement_key, key)
        epsilon = noise.gaussian(jax.random.fold_in(stage_key, current), images.shape)
        return mean + jnp.where(following < 0, 0.0, jnp.sqrt(variance)) * epsilon, None

    return jax.lax.scan(update, images, (times, times - 1))[0]
