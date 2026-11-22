# Development and reproduction

Run commands from the repository root.

## Train

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
uv run --locked python train.py --model tokenizer --epochs 10
uv run --locked python train.py --model transformer --epochs 10
uv run --locked python train.py --model denoiser --epochs 20
```

Train the tokenizer before the transformer, which loads its frozen weights and
encodes MNIST once. The denoiser trains independently. MNIST downloads to ignored
`data/` with archive checksums verified; no TensorFlow or PyTorch is needed.

Train all three models and then regenerate the README grid with:

```sh
sh scripts/reproduce.sh
# Apple Silicon GPU:
METAL=1 sh scripts/reproduce.sh
```

The script uses 10 tokenizer epochs, 10 transformer epochs and 20 denoiser epochs,
then runs the preview generator with those newly saved weights. The grid always
uses CPU, even when training uses Metal.
Arguments are forwarded to each command; `--epochs` overrides these budgets.
Set `PREVIEW_OUTPUT` to save the grid elsewhere; `--output` selects the weights directory.
Other defaults: batch size **128**, Adam learning rate **0.0003**, seed **0**.
Use `--batch-size`, `--learning-rate`, `--seed`, `--output`, or `--help`.

### Apple Silicon GPU

```sh
uv sync --locked --extra metal
uv run --locked --extra metal python train.py --model denoiser --epochs 20
```

The optional [jax-graft](https://github.com/dfm/jax-graft) backend installs only on
macOS ARM64. It is a community Metal backend. JAX selects the accelerator when
available; the tested M4 Mac mini reports `MtlDevice(id=0)`. Keep `--extra metal`
on `uv run` commands to retain the plugin. The base Mac installation uses CPU;
`JAX_PLATFORMS=cpu` forces CPU with the plugin installed. Metal is experimental.

### Quick pipeline check

```sh
METAL=1 PREVIEW_OUTPUT=/tmp/mnist-smoke.png sh scripts/reproduce.sh \
  --epochs 1 --batch-size 4 --limit 8 --output /tmp/mnist-smoke
```

This verifies training, final weight files and sample-grid generation on a tiny
subset, saving all outputs outside the repository. It does not produce useful digits. Omit `METAL=1` for CPU on a Mac.

## Models and objectives

| Model | Architecture and training |
| :-- | :-- |
| Tokenizer | Residual VQ grids 1 × 1, 3 × 3, 7 × 7; 64 codes per level, 32 latent channels. Reconstruction + codebook + 0.25 × commitment loss, using random nonempty prefixes. |
| Transformer | Three blocks, width 128, four attention heads. Predict discrete codes with equal-weight per-scale cross-entropy; earlier conditioning codes have 10% random replacement. |
| Conditional denoiser | U-Net widths 32/64/128, digit and timestep conditioning. Predicts `v`; 10% of training labels are replaced by an unconditional label. |

The tokenizer decoder receives two fixed spatial coordinate channels, following
[CoordConv](https://arxiv.org/abs/1807.03247), so its constant first-level latent
can describe a spatially varying image. The transformer uses one learned table
for all 59 level/position pairs (1 + 9 + 49). It is a small image adaptation of
[Next-Concept Modeling](https://arxiv.org/abs/2610.14162), rather than a reproduction
of that paper's architecture or results. Each scale is predicted in parallel.
Token sampling uses temperature **0.7** and top-p **0.95** at every scale,
including the final-stage refinement. These fixed sampling settings improve digit
consistency with a modest reduction in variation.

Diffusion shares a **200-step cosine schedule** between training and sampling.
For `x_t = a*x_0 + b*epsilon`, the target is `v = a*epsilon - b*x_0` and the clean
image estimate is `x_0 = a*x_t - b*v`. This avoids dividing by a tiny signal
coefficient near pure noise. See [v parameterization](https://arxiv.org/abs/2202.00512).
[Classifier-free guidance](https://arxiv.org/abs/2207.12598) combines unconditional
and digit-conditioned predictions from the same denoiser (default strength 2).
No external classifier or image-gradient calculation is used during generation.

## Training and outputs

Training uses a fixed seed-0 subset of **55,000 images**, reserving 5,000
from the training set and leaving the official test set untouched. With `--limit`,
10% (at least one image) is reserved instead. Each epoch prints its sample-weighted
training loss and elapsed time; training does not generate diagnostic reports.

Only final Flax model weights are saved, one MessagePack file per model:

```text
weights/
  tokenizer.msgpack
  transformer.msgpack
  denoiser.msgpack
```

Retraining replaces the corresponding tracked weight file.
If the tokenizer is retrained, retrain the transformer too, since its code IDs
are tied to that tokenizer. No optimizer state or resume mechanism is stored.

## Sampling and refinement

Two small sampler classes wrap the loaded networks: `Tokens` contains the tokenizer
and transformer; `Diffusion` contains the denoiser. One entry point selects either:

```python
images = sampling.sample(method, models, label=3, seed=42, refinement_seed=0)
# images.shape == (3, 28, 28, 1), values in [0, 1]
```

`method` is `sampling.Method.TOKENS` or `sampling.Method.DIFFUSION`;
`models` maps these methods to loaded sampler instances. The internal
`_sample` function compiles each sampler with Flax NNX. The demo calls the
entry point twice for six images.

The main seed fixes the first two stages. The refinement seed controls only the
last token grid or final 60 diffusion steps. Every call recomputes all three
stages; there is no prefix cache. Refinement therefore preserves the earlier
images without storing intermediate state in the browser.

Fixed inputs and the lockfile reproduce results on the same backend; CPU and
Metal images can differ. Timings include all stages and exclude image rendering and network transfer.
The first generation includes JAX compilation; later requests reuse it.

## Layout

```text
app.py               # Streamlit controls and six generated images
train.py             # train one selected model
models/              # architectures and shared layers
utils/
  data.py            # MNIST, fixed partition, batching and code caching
  noise.py           # shared schedule, v targets and Gaussian noise
  training.py        # trainer classes and shared epoch loop
  sampling.py        # two samplers and one compiled interface
  checkpoints.py     # final Flax MessagePack weights
scripts/reproduce.sh # train all three models, then regenerate the grid
scripts/preview.py   # regenerate the fixed-seed sample grid
assets/samples.png   # generated README grid
tests/               # model, training and Streamlit UI checks
.streamlit/          # shared app theme
```

## Development

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
uv run --locked python -m unittest discover -s tests -v
```

Add `--extra metal` for GPU checks. CI runs on CPU with synthetic inputs, without
downloading MNIST. Python 3.12 and dependencies are pinned through
`.python-version`, `pyproject.toml` and `uv.lock`.


## Regenerate the sample grid

```sh
uv run --locked python -m scripts.preview
```

This generates `assets/samples.png` from the saved weights: digits 0–9, main
seeds **42** and **123**, and refinement seed **0**, for each method. Every
sample is shown; no training, MNIST download or manual image selection is involved.
The script uses CPU even when Metal is installed. Identical weights, dependencies
and CPU execution reproduce the grid.

Use `--weights path/to/weights` or `--output path/to/grid.png` for other paths.
Run it again after training to refresh the README preview.

## Streamlit Community Cloud

Deploy the GitHub repository on
[Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud):

1. Select the repository and the branch containing this app.
2. Set the entry point to `app.py` and Python to **3.12**.
3. Deploy. Community Cloud reads `uv.lock`; no extra requirements file is needed.

The hosted app uses CPU and the three committed weights, without downloading
MNIST or training. Community Cloud has shared resource limits and apps sleep
after inactivity; the first visitor may need to wake the app. A public demo URL
can be added here once deployed. GitHub's README can show a preview and link to
the app, but cannot run these controls inline.

