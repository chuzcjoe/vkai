"""Convert the official MNIST-12 ONNX initializers to a PyTorch state_dict."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = SCRIPT_DIR / "model" / "mnist-12.onnx"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "model" / "mnist12_cnn.pth"
MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx"
)
MODEL_SHA256 = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX Model Zoo MNIST-12 weights to a PyTorch state_dict."
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "vkai-mnist12-weight-converter/1.0"}
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="mnist-12-",
            suffix=".onnx.download",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, temporary_file)

        checksum = file_sha256(temporary_path)
        if checksum != MODEL_SHA256:
            raise RuntimeError(
                "Downloaded model checksum mismatch: "
                f"expected {MODEL_SHA256}, got {checksum}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def resolve_onnx_model(model_path: Path) -> Path:
    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.exists():
        if resolved_path != DEFAULT_ONNX_PATH.resolve():
            raise FileNotFoundError(f"ONNX model does not exist: {resolved_path}")
        print(f"Downloading MNIST-12 model to {resolved_path}")
        download_official_model(resolved_path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"ONNX model is not a file: {resolved_path}")
    if resolved_path == DEFAULT_ONNX_PATH.resolve():
        checksum = file_sha256(resolved_path)
        if checksum != MODEL_SHA256:
            raise RuntimeError(
                "Cached ONNX model checksum mismatch: "
                f"expected {MODEL_SHA256}, got {checksum}"
            )
    return resolved_path


def initializer_array(
    initializers: dict[str, np.ndarray], name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    if name not in initializers:
        raise KeyError(f"Missing ONNX initializer: {name}")
    array = np.asarray(initializers[name], dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {name}: expected {expected_shape}, got {array.shape}"
        )
    return np.ascontiguousarray(array)


def convert_weights(onnx_path: Path, output_path: Path) -> None:
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }

    fc_weight = initializer_array(initializers, "Parameter193", (16, 4, 4, 10))
    state_dict = {
        "conv1.weight": torch.from_numpy(
            initializer_array(initializers, "Parameter5", (8, 1, 5, 5)).copy()
        ),
        "conv1.bias": torch.from_numpy(
            initializer_array(initializers, "Parameter6", (8, 1, 1)).reshape(8).copy()
        ),
        "conv2.weight": torch.from_numpy(
            initializer_array(initializers, "Parameter87", (16, 8, 5, 5)).copy()
        ),
        "conv2.bias": torch.from_numpy(
            initializer_array(initializers, "Parameter88", (16, 1, 1))
            .reshape(16)
            .copy()
        ),
        "fc.weight": torch.from_numpy(fc_weight.reshape(256, 10).T.copy()),
        "fc.bias": torch.from_numpy(
            initializer_array(initializers, "Parameter194", (1, 10)).reshape(10).copy()
        ),
    }

    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, resolved_output)
    print(f"Source ONNX: {onnx_path}")
    print(f"PyTorch weights: {resolved_output}")
    for name, tensor in state_dict.items():
        print(f"  {name}: shape={list(tensor.shape)}")


def main() -> None:
    args = parse_args()
    onnx_path = resolve_onnx_model(args.onnx)
    convert_weights(onnx_path, args.output)


if __name__ == "__main__":
    main()
