"""Pipeline invariants and short updates, without a dataset download."""

import gzip
import hashlib
import pathlib
import struct
import tempfile
import unittest
import unittest.mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import train
from models import tokenizer, transformer
from utils import checkpoints, data, noise, sampling, training


class Tiny(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


class MeanTrainer(training.Trainer):
    def loss(self, model: Tiny, batch: data.Batch, key: jax.Array) -> jax.Array:
        return jnp.mean(batch[0])


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = jax.random.uniform(jax.random.key(0), (2, 28, 28, 1))
        self.labels = jnp.array([2, 7], dtype=jnp.int32)

    def test_partition_is_fixed_aligned_and_disjoint(self) -> None:
        ids = np.arange(100)
        train, validation = data.partition((ids, ids * 3))
        repeated_train, repeated_validation = data.partition((ids, ids * 3))
        self.assertEqual(len(train[0]), 90)
        self.assertEqual(len(validation[0]), 10)
        self.assertEqual(len(set(train[0]) & set(validation[0])), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate((train[0], validation[0]))), ids)
        for first, second in zip(
            train + validation,
            repeated_train + repeated_validation,
            strict=True,
        ):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(train[1], train[0] * 3)
        np.testing.assert_array_equal(validation[1], validation[0] * 3)
        full_train, full_validation = data.partition((np.arange(60000),))
        self.assertEqual(len(full_train[0]), 55000)
        self.assertEqual(len(full_validation[0]), 5000)

    def test_mnist_loading_and_archive_validation(self) -> None:
        images = np.arange(2 * 28 * 28, dtype=np.int32).astype(np.uint8).reshape(2, 28, 28, 1)
        labels = np.array([3, 8], dtype=np.uint8)
        archives = {
            "train-images-idx3-ubyte.gz": gzip.compress(
                struct.pack(">4I", 2051, 2, 28, 28) + images.tobytes(),
            ),
            "train-labels-idx1-ubyte.gz": gzip.compress(
                struct.pack(">2I", 2049, 2) + labels.tobytes(),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            checksums = {}
            for name, contents in archives.items():
                (path / name).write_bytes(contents)
                checksums[name] = hashlib.md5(contents).hexdigest()
            with unittest.mock.patch.dict(data.FILES, checksums):
                with unittest.mock.patch("urllib.request.urlretrieve") as download:
                    loaded, targets = data.load(path, limit=1)
                    download.assert_not_called()
                self.assertEqual(loaded.dtype, np.float32)
                self.assertEqual(targets.dtype, np.int32)
                np.testing.assert_allclose(loaded, images[:1] / 255)
                np.testing.assert_array_equal(targets, labels[:1])
                (path / "train-labels-idx1-ubyte.gz").write_bytes(b"invalid")
                with self.assertRaisesRegex(ValueError, "Invalid MNIST"):
                    data.load(path)

    def test_batches_are_aligned_complete_and_repeatable(self) -> None:
        labels = np.arange(5, dtype=np.int32)
        arrays = (labels[:, None] * 10, labels)
        first = list(data.batches(arrays, 2, np.random.default_rng(0)))
        second = list(data.batches(arrays, 2, np.random.default_rng(0)))
        self.assertEqual([len(batch[0]) for batch in first], [2, 2, 1])
        np.testing.assert_array_equal(
            np.sort(np.concatenate([batch[1] for batch in first])),
            labels,
        )
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a[0].squeeze(-1), a[1] * 10)
            np.testing.assert_array_equal(a[1], b[1])

    def test_shared_loop_weights_partial_batches_correctly(self) -> None:
        trainer = MeanTrainer(Tiny(rngs=nnx.Rngs(0)), learning_rate=1e-3)
        with unittest.mock.patch("builtins.print") as output:
            training.fit(
                trainer,
                (np.arange(5, dtype=np.float32),),
                epochs=2,
                batch_size=2,
                seed=0,
            )
        self.assertEqual(output.call_count, 2)
        for call in output.call_args_list:
            self.assertIn("loss=2.00000", call.args[0])

    def test_noise_is_seeded_and_finest_timestep_is_nearly_gaussian(self) -> None:
        self.assertTrue(bool(jnp.all(jnp.diff(noise.ALPHA_BARS) < 0)))
        self.assertLess(float(noise.ALPHA_BARS[-1]), 1e-6)
        first = noise.sample(jax.random.key(1), self.images)
        second = noise.sample(jax.random.key(1), self.images)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)
        timesteps = jnp.full((2,), noise.STEPS - 1, dtype=jnp.int32)
        noisy = noise.add(self.images, jnp.ones_like(self.images), timesteps)
        np.testing.assert_allclose(noisy, 1, atol=1e-3)

    def test_three_updates_and_frozen_tokenizer(self) -> None:
        codec = tokenizer.Tokenizer(rngs=nnx.Rngs(4))
        codes = codec.encode(self.images)
        frozen_before = [np.array(x) for x in jax.tree.leaves(nnx.state(codec, nnx.Param))]
        token_trainer = training.TransformerTrainer(seed=1, learning_rate=3e-4)
        token_trainer.codec = codec
        cases = [
            (training.TokenizerTrainer(0, 3e-4), (self.images, self.labels)),
            (token_trainer, (*codes, self.labels)),
            (training.DenoiserTrainer(2, 3e-4), (self.images, self.labels)),
        ]
        for trainer, batch in cases:
            with self.subTest(model=type(trainer.model).__name__):
                before = [np.array(x) for x in jax.tree.leaves(nnx.state(trainer.model, nnx.Param))]
                loss = trainer.step(batch, jax.random.key(5))
                self.assertTrue(np.isfinite(float(loss)))
                after = jax.tree.leaves(nnx.state(trainer.model, nnx.Param))
                self.assertTrue(
                    any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True)),
                )
                self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in after))
        frozen_after = jax.tree.leaves(nnx.state(codec, nnx.Param))
        for a, b in zip(frozen_before, frozen_after, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_code_cache_keeps_last_partial_batch(self) -> None:
        codec = tokenizer.Tokenizer(rngs=nnx.Rngs(4))
        images = np.asarray(jnp.concatenate((self.images, self.images[:1])))
        cached = data.encode(codec, images, batch_size=2)
        direct = codec.encode(jnp.asarray(images))
        for a, b in zip(cached, direct, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_flax_checkpoint_round_trip_across_devices(self) -> None:
        model = Tiny(rngs=nnx.Rngs(0))
        expected = jax.device_get(nnx.state(model, nnx.Param))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tiny.msgpack"
            checkpoints.save(model, path)
            with jax.default_device(jax.devices("cpu")[0]):
                restored = Tiny(rngs=nnx.Rngs(9))
                checkpoints.load(restored, path)
            for a, b in zip(
                jax.tree.leaves(expected),
                jax.tree.leaves(nnx.state(restored, nnx.Param)),
                strict=True,
            ):
                np.testing.assert_array_equal(a, b)
                self.assertEqual(next(iter(b.devices())).platform, "cpu")

    def test_training_selects_one_model(self) -> None:
        for name in ("tokenizer", "transformer", "denoiser"):
            cfg = train.Config.model_validate({"model": name})
            self.assertIs(cfg.model, training.ModelKind(name))
        for name in ("all", "tokens", "diffusion", "classifier"):
            with self.assertRaises(ValueError):
                train.Config.model_validate({"model": name})

    def test_prepare_preserves_inputs_for_image_models(self) -> None:
        arrays = (np.asarray(self.images), np.asarray(self.labels))
        with tempfile.TemporaryDirectory() as directory:
            for kind in (
                training.ModelKind.TOKENIZER,
                training.ModelKind.DENOISER,
            ):
                trainer = training.create(kind, seed=0, learning_rate=3e-4)
                prepared = trainer.prepare(arrays, weights=pathlib.Path(directory), batch_size=2)
                self.assertIs(prepared, arrays)
                value = trainer.step(tuple(jnp.asarray(x) for x in prepared), jax.random.key(1))
                self.assertTrue(np.isfinite(float(value)))

    def test_transformer_trainer_loads_its_frozen_tokenizer(self) -> None:
        images, labels = np.asarray(self.images), np.asarray(self.labels)
        codec = tokenizer.Tokenizer(rngs=nnx.Rngs(4))
        expected = data.encode(codec, images, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            weights = pathlib.Path(directory)
            checkpoints.save(codec, weights / "tokenizer.msgpack")
            trainer = training.create(training.ModelKind.TRANSFORMER, seed=0, learning_rate=3e-4)
            arrays = trainer.prepare((images, labels), weights=weights, batch_size=2)
            self.assertIsInstance(trainer.model, transformer.Transformer)
            self.assertIs(arrays[-1], labels)
            for actual, target in zip(arrays[:-1], expected, strict=True):
                np.testing.assert_array_equal(actual, target)
            value = trainer.step(tuple(jnp.asarray(x) for x in arrays), jax.random.key(1))
            self.assertTrue(np.isfinite(float(value)))

    def test_velocity_recovers_clean_image_at_all_noise_levels(self) -> None:
        timesteps = jnp.array([0, noise.STEPS - 1])
        epsilon = jax.random.normal(jax.random.key(2), self.images.shape)
        clean = self.images * 2 - 1
        noisy = noise.add(clean, epsilon, timesteps)
        velocity = noise.velocity(clean, epsilon, timesteps)
        alpha = noise.ALPHA_BARS[timesteps][:, None, None, None]
        recovered = jnp.sqrt(alpha) * noisy - jnp.sqrt(1 - alpha) * velocity
        np.testing.assert_allclose(recovered, clean, atol=1e-6)

    def test_token_nucleus_sampling_excludes_the_tail(self) -> None:
        logits = jnp.broadcast_to(jnp.log(jnp.array([0.97, 0.02, 0.01])), (1000, 3))
        samples = jax.jit(sampling._sample_codes)(jax.random.key(0), logits)
        np.testing.assert_array_equal(samples, 0)

    def test_token_nucleus_keeps_the_threshold_crossing_code(self) -> None:
        logits = jnp.broadcast_to(jnp.log(jnp.array([0.7, 0.25, 0.05])), (1000, 3))
        samples = jax.jit(sampling._sample_codes)(jax.random.key(0), logits)
        self.assertFalse(bool(jnp.any(samples == 2)))
        self.assertTrue(bool(jnp.any(samples == 1)))
        np.testing.assert_array_equal(
            samples,
            jax.jit(sampling._sample_codes)(jax.random.key(0), logits),
        )


if __name__ == "__main__":
    unittest.main()
