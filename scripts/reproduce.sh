#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

if [ "${METAL:-0}" = "1" ]; then
    uv sync --locked --extra metal
else
    uv sync --locked
fi

weights=weights
previous=
for argument do
    if [ "$previous" = "--output" ]; then weights=$argument; fi
    case "$argument" in
        --output=*) weights=${argument#--output=} ;;
    esac
    previous=$argument
done

for model in tokenizer transformer denoiser; do
    epochs=10
    if [ "$model" = "denoiser" ]; then epochs=20; fi
    .venv/bin/python train.py --model "$model" --epochs "$epochs" --output "$weights" "$@"
done

.venv/bin/python -m scripts.preview --weights "$weights" --output "${PREVIEW_OUTPUT:-assets/samples.png}"
