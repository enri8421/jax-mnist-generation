"""Train one model and save its final weights."""

import pathlib

import jax
import pydantic
import pydantic_settings

from utils import checkpoints, data, training


class Config(pydantic_settings.BaseSettings):
    """Shared training settings; architecture and algorithm constants stay in code."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_kebab_case=True,
        cli_enforce_required=True,
    )

    model: training.ModelKind
    epochs: int = pydantic.Field(default=10, ge=1)
    batch_size: int = pydantic.Field(default=128, ge=1)
    learning_rate: float = pydantic.Field(default=3e-4, gt=0)
    seed: int = pydantic.Field(default=0, ge=0, le=2**32 - 1)
    limit: int | None = pydantic.Field(default=None, ge=2)
    output: pathlib.Path = pathlib.Path("weights")


def train(cfg: Config) -> None:
    arrays, _ = data.partition(data.load(limit=cfg.limit))
    trainer = training.create(cfg.model, seed=cfg.seed, learning_rate=cfg.learning_rate)
    arrays = trainer.prepare(arrays, weights=cfg.output, batch_size=cfg.batch_size)
    print(f"Backend: {jax.devices()[0]}; training samples: {len(arrays[0])}", flush=True)
    training.fit(
        trainer,
        arrays,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    destination = cfg.output / f"{cfg.model}.msgpack"
    checkpoints.save(trainer.model, destination)
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    train(pydantic_settings.CliApp.run(Config))
