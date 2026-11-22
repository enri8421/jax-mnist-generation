"""Two digit generators, presented with Streamlit."""

import os
import pathlib
import secrets
import shlex
import threading
import time

import jax
import numpy as np
import numpy.typing as npt
import pydantic
import streamlit as st

from utils import checkpoints, sampling

WEIGHTS = pathlib.Path(os.environ.get("MNIST_WEIGHTS", pathlib.Path(__file__).parent / "weights"))
MAX_SEED = 2**31 - 1


class Request(pydantic.BaseModel):
    digit: int = pydantic.Field(ge=0, le=9)
    seed: int = pydantic.Field(ge=0, le=MAX_SEED)
    refinement_seed: int = pydantic.Field(ge=0, le=MAX_SEED)


class Generator:
    def __init__(self, weights: pathlib.Path) -> None:
        names = ("tokenizer", "transformer", "denoiser")
        missing = [
            str(weights / f"{name}.msgpack")
            for name in names
            if not (weights / f"{name}.msgpack").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing model weights: {', '.join(missing)}. "
                f"Run: sh scripts/reproduce.sh --output {shlex.quote(str(weights))}",
            )
        tokens = sampling.Tokens()
        diffusion = sampling.Diffusion()
        self.models: dict[sampling.Method, sampling.Tokens | sampling.Diffusion] = {
            sampling.Method.TOKENS: tokens,
            sampling.Method.DIFFUSION: diffusion,
        }
        self.lock = threading.Lock()
        for name, model in zip(names, (tokens.codec, tokens.model, diffusion.model), strict=True):
            checkpoints.load(model, weights / f"{name}.msgpack")

    def generate(self, request: Request) -> tuple[npt.NDArray[np.float32], list[int]]:
        images, milliseconds = [], []
        with self.lock:
            for method in sampling.Method:
                started = time.perf_counter()
                stages = sampling.sample(
                    method,
                    self.models,
                    label=request.digit,
                    seed=request.seed,
                    refinement_seed=request.refinement_seed,
                )
                stages.block_until_ready()
                milliseconds.append(round((time.perf_counter() - started) * 1000))
                images.append(np.asarray(stages[..., 0]))
        return np.stack(images), milliseconds  # (2, 3, 28, 28), one timing per method.


@st.cache_resource(show_spinner="Loading the three trained models…")
def load_generator(weights: pathlib.Path) -> Generator:
    return Generator(weights)


def randomize(refine: bool = False) -> None:
    """Change only the refinement seed when refining."""
    for key in ("refinement_seed",) if refine else ("seed", "refinement_seed"):
        current = st.session_state[key]
        st.session_state[key] = (current + 1 + secrets.randbelow(MAX_SEED)) % (MAX_SEED + 1)


def main() -> None:
    st.set_page_config(page_title="Two ways to make a digit", page_icon="✍️", layout="wide")
    st.caption("JAX / FLAX · A SMALL GENERATIVE EXPERIMENT")
    st.title("Two ways to make a digit")
    st.write("Choose a number. Generate it with tokens or diffusion. Refine the final stage.")

    st.session_state.setdefault("seed", 42)
    st.session_state.setdefault("refinement_seed", 0)
    with st.container(border=True):
        st.pills("Digit", range(10), default=3, key="digit", required=True)
        seed, refinement_seed = st.columns(2)
        seed.number_input("Seed", min_value=0, max_value=MAX_SEED, step=1, key="seed")
        refinement_seed.number_input(
            "Refinement seed",
            min_value=0,
            max_value=MAX_SEED,
            step=1,
            key="refinement_seed",
        )
        generate, refine = st.columns(2)
        generate.button("Regenerate both", type="primary", on_click=randomize, width="stretch")
        refine.button("Refine again", on_click=randomize, args=(True,), width="stretch")
        st.caption(
            "The main seed fixes the first two images. The refinement seed changes only the last."
        )

    try:
        generator = load_generator(WEIGHTS)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    request = Request(
        digit=st.session_state.digit,
        seed=st.session_state.seed,
        refinement_seed=st.session_state.refinement_seed,
    )
    if st.session_state.get("request") != request:
        with st.spinner("Generating both methods… First use includes JAX compilation."):
            images, milliseconds = generator.generate(request)
        st.session_state.request = request
        st.session_state.images = images
        st.session_state.milliseconds = milliseconds

    for index, (column, title, description, captions) in enumerate(
        zip(
            st.columns(2),
            ("Residual VQ + Transformer", "Conditional diffusion"),
            ("Three residual grids build the digit.", "Two hundred steps turn noise into a digit."),
            (("1 × 1", "3 × 3", "7 × 7"), ("0 steps", "140 steps", "200 steps")),
            strict=True,
        )
    ):
        with column.container(border=True):
            st.subheader(title)
            st.caption(description)
            with st.container(
                horizontal=True, wrap=False, gap="xsmall", horizontal_alignment="center"
            ):
                for level, caption in enumerate(captions):
                    with st.container(width=96):
                        st.image(
                            st.session_state.images[index, level],
                            caption=caption,
                            alt=f"{title}, digit {request.digit}, {caption}",
                            width="stretch",
                            clamp=True,
                        )
            st.caption(f"{st.session_state.milliseconds[index]} ms · generation time")
            with st.expander("How does refinement work?"):
                st.write(
                    "Every call recomputes all stages with the same main seed. "
                    "Only the final token grid or last 60 diffusion steps use the refinement seed. "
                    "Earlier images stay the same; final strokes can change.",
                )

    st.caption(
        f"28 × 28 pixels · {jax.devices()[0]} · Fixed seeds are repeatable on the same backend."
    )


if __name__ == "__main__":
    main()
