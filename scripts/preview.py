"""Generate the README grid from saved weights; no training or dataset needed."""

import pathlib

import jax
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import pydantic_settings

from utils import checkpoints, sampling

SEEDS = (42, 123)
REFINEMENT_SEED = 0


class Config(pydantic_settings.BaseSettings):
    weights: pathlib.Path = pathlib.Path("weights")
    output: pathlib.Path = pathlib.Path("assets/samples.png")


def main(cfg: Config) -> None:
    canvas = PIL.Image.new("RGB", (1196, 586), "#f7f9f7")
    draw = PIL.ImageDraw.Draw(canvas)
    heading = PIL.ImageFont.load_default(size=28)
    label = PIL.ImageFont.load_default(size=18)
    small = PIL.ImageFont.load_default(size=14)
    draw.text((32, 26), "Two ways to make a digit", font=heading, fill="#172823")
    draw.text(
        (32, 68), "Final generated samples. Two fixed seeds per method.", font=small, fill="#62756c"
    )
    for digit in range(10):
        draw.text((258 + digit * 96, 110), str(digit), anchor="mm", font=label, fill="#172823")

    # Use CPU for the published grid, even when the Metal extra is installed.
    with jax.default_device(jax.devices("cpu")[0]):
        tokens, diffusion = sampling.Tokens(), sampling.Diffusion()
        models: dict[sampling.Method, sampling.Tokens | sampling.Diffusion] = {
            sampling.Method.TOKENS: tokens,
            sampling.Method.DIFFUSION: diffusion,
        }
        for name, model in (
            ("tokenizer", tokens.codec),
            ("transformer", tokens.model),
            ("denoiser", diffusion.model),
        ):
            checkpoints.load(model, cfg.weights / f"{name}.msgpack")

        for method_index, method in enumerate(sampling.Method):
            for seed_index, seed in enumerate(SEEDS):
                y = 132 + (method_index * len(SEEDS) + seed_index) * 104
                name = "Transformer" if method == sampling.Method.TOKENS else "Diffusion"
                draw.text((32, y + 24), name, font=label, fill="#172823")
                draw.text((32, y + 52), f"Seed {seed}", font=small, fill="#62756c")
                for digit in range(10):
                    stages = sampling.sample(method, models, digit, seed, REFINEMENT_SEED)
                    pixels = np.rint(np.asarray(stages[-1, :, :, 0]) * 255).astype(np.uint8)
                    image = PIL.Image.fromarray(pixels).resize(
                        (84, 84), PIL.Image.Resampling.NEAREST
                    )
                    canvas.paste(image, (216 + digit * 96, y))
                print(f"{name}: seed {seed}, digits 0-9", flush=True)

    draw.line((32, 330, 1164, 330), fill="#dbe4de", width=1)
    draw.text(
        (32, 550),
        f"Refinement seed: {REFINEMENT_SEED}  |  CPU  |  Saved model weights",
        font=small,
        fill="#62756c",
    )
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(cfg.output)
    print(f"Saved {cfg.output}", flush=True)


if __name__ == "__main__":
    main(pydantic_settings.CliApp.run(Config))
