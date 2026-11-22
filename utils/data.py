"""MNIST loading and shared batching, without TensorFlow or PyTorch."""

import collections.abc
import gzip
import hashlib
import pathlib
import struct
import typing
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import nnx

from models import tokenizer

type Arrays = tuple[npt.NDArray, ...]
type Batch = tuple[jax.Array, ...]

# Published MNIST archive checksums, also used by torchvision.datasets.MNIST.
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load(
    directory: pathlib.Path = pathlib.Path("data"),
    *,
    split: typing.Literal["train", "test"] = "train",
    limit: int | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Download/cache MNIST; return [0, 1] NHWC images and integer labels."""
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "train" if split == "train" else "t10k"
    contents = []
    for suffix in ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz"):
        name = f"{prefix}-{suffix}"
        path = directory / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(
                f"https://ossci-datasets.s3.amazonaws.com/mnist/{name}",
                temporary,
            )
            temporary.replace(path)
        if hashlib.md5(path.read_bytes()).hexdigest() != FILES[name]:
            raise ValueError(f"Invalid MNIST archive; remove {path} and retry")
        contents.append(gzip.decompress(path.read_bytes()))
    magic, count, height, width = struct.unpack(">4I", contents[0][:16])
    label_magic, label_count = struct.unpack(">2I", contents[1][:8])
    if (magic, height, width, label_magic, label_count) != (2051, 28, 28, 2049, count):
        raise ValueError("Unexpected MNIST headers")
    images = np.frombuffer(contents[0], np.uint8, offset=16).reshape(count, 28, 28, 1)
    labels = np.frombuffer(contents[1], np.uint8, offset=8)
    return images[:limit].astype(np.float32) / np.float32(255), labels[:limit].astype(np.int32)


def partition(arrays: Arrays) -> tuple[Arrays, Arrays]:
    """Reserve a fixed validation subset, independent of the training seed."""
    count = len(arrays[0])
    if count < 2 or any(len(array) != count for array in arrays):
        raise ValueError("Expected at least two aligned samples")
    order = np.random.default_rng(0).permutation(count)
    validation_size = max(1, min(5000, count // 10))
    validation, training = order[:validation_size], order[validation_size:]
    return tuple(array[training] for array in arrays), tuple(array[validation] for array in arrays)


def batches(
    arrays: Arrays,
    batch_size: int,
    rng: np.random.Generator,
) -> collections.abc.Iterator[Batch]:
    """Shuffle aligned arrays together, retaining the final partial batch."""
    count = len(arrays[0])
    if batch_size < 1 or count == 0 or any(len(array) != count for array in arrays):
        raise ValueError("Expected nonempty aligned arrays and a positive batch size")
    order = rng.permutation(count)
    for start in range(0, count, batch_size):
        indices = order[start : start + batch_size]
        yield tuple(jnp.asarray(array[indices]) for array in arrays)


def encode(
    codec: tokenizer.Tokenizer,
    images: npt.NDArray[np.float32],
    batch_size: int,
) -> Arrays:
    """Encode once after tokenizer training; keep the code cache in host memory."""
    # Keep both outputs: Metal can miscompile the fused codes-only quantizer path.
    encode_batch = nnx.jit(lambda model, batch: model.quantize(model.encoder(batch)))
    encoded = [
        jax.device_get(encode_batch(codec, jnp.asarray(images[start : start + batch_size]))[1])
        for start in range(0, len(images), batch_size)
    ]
    return tuple(np.concatenate(level) for level in zip(*encoded, strict=True))
