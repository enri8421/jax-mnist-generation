"""Seeded sampling and demo input contracts, without training."""

import pathlib
import tempfile
import unittest
import unittest.mock

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import streamlit.testing.v1 as testing

import app
from utils import noise, sampling


class DemoTests(unittest.TestCase):
    def test_seeded_gaussian_noise_is_repeatable(self) -> None:
        first = noise.gaussian(jax.random.key(42), (1024,))
        np.testing.assert_array_equal(first, noise.gaussian(jax.random.key(42), (1024,)))
        self.assertEqual(first.shape, (1024,))
        self.assertTrue(np.isfinite(np.asarray(first)).all())
        self.assertFalse(np.array_equal(first, noise.gaussian(jax.random.key(43), (1024,))))

    def test_request_rejects_invalid_controls(self) -> None:
        valid = {"digit": 3, "seed": 42, "refinement_seed": 0}
        self.assertEqual(app.Request.model_validate(valid).digit, 3)
        for changes in (
            {"digit": 10},
            {"seed": -1},
            {"seed": 2**31},
        ):
            with self.assertRaises(pydantic.ValidationError):
                app.Request.model_validate(valid | changes)
        for refinement_seed in (-1, 2**31):
            with self.assertRaises(pydantic.ValidationError):
                app.Request(digit=3, seed=42, refinement_seed=refinement_seed)

    def test_missing_weights_explain_how_to_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = pathlib.Path(directory)
            (weights / "tokenizer.msgpack").touch()
            with unittest.mock.patch("app.checkpoints.load") as load:
                with self.assertRaisesRegex(FileNotFoundError, "sh scripts/reproduce.sh"):
                    app.Generator(weights)
            load.assert_not_called()


class UnifiedDemoTests(unittest.TestCase):
    def test_both_methods_return_three_repeatable_stages(self) -> None:
        models: dict[sampling.Method, sampling.Tokens | sampling.Diffusion] = {
            sampling.Method.TOKENS: sampling.Tokens(),
            sampling.Method.DIFFUSION: sampling.Diffusion(),
        }
        for method in sampling.Method:
            with self.subTest(method=method):
                first = sampling.sample(method, models, 3, 42, 0)
                repeated = sampling.sample(method, models, 3, 42, 0)
                refined = sampling.sample(method, models, 3, 42, 1)
                changed = sampling.sample(method, models, 3, 43, 0)
                self.assertEqual(first.shape, (3, 28, 28, 1))
                self.assertTrue(np.isfinite(np.asarray(first)).all())
                self.assertTrue(bool(jnp.all((first >= 0) & (first <= 1))))
                np.testing.assert_array_equal(first, repeated)
                np.testing.assert_array_equal(first[:2], refined[:2])
                self.assertFalse(np.array_equal(first[2], refined[2]))
                self.assertFalse(np.array_equal(first[:2], changed[:2]))

    def test_demo_calls_each_method_with_the_same_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            weights = pathlib.Path(temporary)
            for name in ("tokenizer", "transformer", "denoiser"):
                (weights / f"{name}.msgpack").touch()
            with unittest.mock.patch("app.checkpoints.load"):
                generator = app.Generator(weights)
            with unittest.mock.patch(
                "app.sampling.sample",
                return_value=jnp.zeros((3, 28, 28, 1)),
            ) as sample:
                images, milliseconds = generator.generate(
                    app.Request(digit=3, seed=42, refinement_seed=1)
                )
            self.assertEqual(images.shape, (2, 3, 28, 28))
            self.assertEqual(len(milliseconds), 2)
            self.assertEqual(sample.call_count, 2)
            for call, method in zip(sample.call_args_list, sampling.Method, strict=True):
                self.assertEqual(call.args, (method, generator.models))
                self.assertEqual(call.kwargs, {"label": 3, "seed": 42, "refinement_seed": 1})


class StreamlitTests(unittest.TestCase):
    def test_controls_keep_the_main_seed_when_refining(self) -> None:
        def render():
            import app

            app.main()

        generator = unittest.mock.Mock()
        generator.generate.return_value = (np.zeros((2, 3, 28, 28), dtype=np.float32), [1, 2])
        with unittest.mock.patch("app.load_generator", return_value=generator):
            page = testing.AppTest.from_function(render).run()
            self.assertFalse(page.exception)
            self.assertEqual(len(page.get("image")), 6)
            self.assertEqual(generator.generate.call_count, 1)
            page.run()
            self.assertEqual(generator.generate.call_count, 1)
            page.button[1].click().run()
            self.assertFalse(page.exception)
            refined = generator.generate.call_args.args[0]
            self.assertEqual(refined.seed, 42)
            self.assertNotEqual(refined.refinement_seed, 0)
            page.button[0].click().run()
            self.assertFalse(page.exception)
            regenerated = generator.generate.call_args.args[0]
            self.assertNotEqual(regenerated.seed, refined.seed)
            self.assertNotEqual(regenerated.refinement_seed, refined.refinement_seed)
            page.number_input(key="seed").set_value(123).run()
            self.assertEqual(generator.generate.call_args.args[0].seed, 123)

    def test_missing_weights_show_an_actionable_error(self) -> None:
        def render():
            import app

            app.main()

        with unittest.mock.patch(
            "app.load_generator",
            side_effect=FileNotFoundError("Run sh scripts/reproduce.sh first"),
        ):
            page = testing.AppTest.from_function(render).run()
        self.assertFalse(page.exception)
        self.assertIn("sh scripts/reproduce.sh", page.error[0].value)
