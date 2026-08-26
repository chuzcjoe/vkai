"""Run pretrained MNIST-12 CNN inference and export every ONNX node output.

All ``.bin`` files are raw, headerless, little-endian float32 arrays. Tensor
shapes, ONNX node attributes, preprocessing details, and the prediction are
recorded in ``metadata.json``.

Examples:
    python python/mnist12_cnn/test_mnist12_cnn.py --image digit.png
    python python/mnist12_cnn/test_mnist12_cnn.py --image digit.png --invert
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "model" / "mnist-12.onnx"
MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx"
)
MODEL_SHA256 = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained MNIST-12 CNN on one image and export float32 "
            "reference binaries for every ONNX node."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image. It is converted to grayscale and resized to 28x28.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the image, useful for a dark digit on a light background.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "ONNX model path. The official model is downloaded automatically "
            "when the default path is missing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "test_data",
        help="Directory for exported .bin files and metadata.json.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "vkai-mnist12-reference-export/1.0"}
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

        downloaded_sha256 = file_sha256(temporary_path)
        if downloaded_sha256 != MODEL_SHA256:
            raise RuntimeError(
                "Downloaded model checksum mismatch: "
                f"expected {MODEL_SHA256}, got {downloaded_sha256}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def resolve_model(model_path: Path) -> tuple[Path, str]:
    resolved_path = model_path.expanduser().resolve()
    default_path = DEFAULT_MODEL_PATH.resolve()

    if not resolved_path.exists():
        if resolved_path != default_path:
            raise FileNotFoundError(f"ONNX model does not exist: {resolved_path}")
        print(f"Downloading MNIST-12 model to {resolved_path}")
        download_official_model(resolved_path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"ONNX model is not a file: {resolved_path}")

    checksum = file_sha256(resolved_path)
    if resolved_path == default_path and checksum != MODEL_SHA256:
        raise RuntimeError(
            "The cached official MNIST-12 model has an unexpected checksum: "
            f"{checksum}. Remove it and run the script again to re-download it."
        )
    return resolved_path, checksum


def load_input(image_path: Path, invert: bool) -> tuple[np.ndarray, dict[str, Any]]:
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {resolved_path}")

    with Image.open(resolved_path) as opened_image:
        original_mode = opened_image.mode
        original_size = list(opened_image.size)
        image = opened_image.convert("L").resize(
            (28, 28), Image.Resampling.BILINEAR
        )
        if invert:
            image = ImageOps.invert(image)
        image_array = np.asarray(image, dtype=np.float32) / np.float32(255.0)

    input_array = np.ascontiguousarray(image_array.reshape(1, 1, 28, 28))
    metadata = {
        "source": str(resolved_path),
        "original_mode": original_mode,
        "original_size_wh": original_size,
        "converted_mode": "L",
        "resize_size_wh": [28, 28],
        "interpolation": "bilinear",
        "inverted": invert,
        "value_range": [0.0, 1.0],
        "normalization": "pixel / 255.0; no mean/std normalization",
        "expected_foreground": "white digit on black background",
        "output_layout": "NCHW",
    }
    return input_array, metadata


def run_inference(
    model: onnx.ModelProto, input_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if len(model.graph.input) != 1:
        raise ValueError(f"Expected one model input, got {len(model.graph.input)}")
    if len(model.graph.output) != 1:
        raise ValueError(f"Expected one model output, got {len(model.graph.output)}")

    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    evaluator = ReferenceEvaluator(model)
    results = evaluator.run(None, {input_name: input_array}, intermediate=True)

    layers = []
    execution_index = 0
    for node in model.graph.node:
        for output_index, node_output_name in enumerate(node.output):
            if not node_output_name:
                continue
            execution_index += 1
            layers.append(
                {
                    "execution_index": execution_index,
                    "node": node.name or f"unnamed_{execution_index}",
                    "op_type": node.op_type,
                    "output_index": output_index,
                    "output_name": node_output_name,
                    "inputs": list(node.input),
                    "attributes": {
                        attribute.name: json_compatible(
                            helper.get_attribute_value(attribute)
                        )
                        for attribute in node.attribute
                    },
                    "array": np.asarray(results[node_output_name]),
                }
            )

    logits = np.asarray(results[output_name], dtype=np.float32)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return logits, probabilities.astype(np.float32), layers


def json_compatible(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_layout(array: np.ndarray) -> str:
    if array.ndim == 4:
        return "NCHW"
    if array.ndim == 2:
        return "NC"
    return f"{array.ndim}D"


def export_array(array: np.ndarray, path: Path) -> dict[str, Any]:
    float_array = np.ascontiguousarray(array, dtype="<f4")
    float_array.tofile(path)
    return {
        "file": path.name,
        "dtype": "float32-little-endian",
        "shape": list(float_array.shape),
        "layout": tensor_layout(float_array),
        "elements": int(float_array.size),
        "bytes": int(float_array.nbytes),
        "min": float(float_array.min()),
        "max": float(float_array.max()),
    }


def safe_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).replace(".", "_")


def export_reference_data(
    output_dir: Path,
    input_array: np.ndarray,
    layers: list[dict[str, Any]],
    logits: np.ndarray,
    probabilities: np.ndarray,
    preprocessing_metadata: dict[str, Any],
    model: onnx.ModelProto,
    model_path: Path,
    model_checksum: str,
) -> Path:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    input_metadata = export_array(input_array, resolved_output_dir / "000_input.bin")
    layer_metadata = []
    for layer in layers:
        filename = (
            f"{layer['execution_index']:03d}_"
            f"{safe_filename_component(str(layer['node']))}_"
            f"{safe_filename_component(str(layer['op_type']))}_output.bin"
        )
        tensor_metadata = export_array(
            layer["array"], resolved_output_dir / filename
        )
        layer_metadata.append(
            {
                key: value
                for key, value in layer.items()
                if key != "array"
            }
            | tensor_metadata
        )

    logits_metadata = export_array(logits, resolved_output_dir / "model_logits.bin")
    probabilities_metadata = export_array(
        probabilities, resolved_output_dir / "softmax_output.bin"
    )

    sorted_indices = np.argsort(probabilities[0])[::-1]
    top5 = [
        {
            "class_index": int(class_index),
            "label": str(int(class_index)),
            "probability": float(probabilities[0, class_index]),
        }
        for class_index in sorted_indices[:5]
    ]
    metadata = {
        "format": "Raw headerless little-endian IEEE-754 float32",
        "model": "ONNX Model Zoo MNIST-12 CNN",
        "model_path": str(model_path),
        "model_url": MODEL_URL,
        "model_sha256": model_checksum,
        "onnx_ir_version": model.ir_version,
        "opsets": [
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
            for opset in model.opset_import
        ],
        "runtime": "onnx.reference.ReferenceEvaluator",
        "preprocessing": preprocessing_metadata,
        "prediction": top5[0],
        "top5": top5,
        "input": input_metadata,
        "layers": layer_metadata,
        "model_logits": logits_metadata,
        "softmax_output": probabilities_metadata,
        "notes": [
            "Node outputs are ordered by ONNX graph execution order.",
            "Every non-empty output of every ONNX node is exported.",
            "model_logits.bin duplicates the final graph output for convenience.",
            "Softmax is applied after the model because MNIST-12 outputs logits.",
        ],
    }

    metadata_path = resolved_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    model_path, model_checksum = resolve_model(args.model)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    input_array, preprocessing_metadata = load_input(args.image, args.invert)
    logits, probabilities, layers = run_inference(model, input_array)
    metadata_path = export_reference_data(
        output_dir=args.output_dir,
        input_array=input_array,
        layers=layers,
        logits=logits,
        probabilities=probabilities,
        preprocessing_metadata=preprocessing_metadata,
        model=model,
        model_path=model_path,
        model_checksum=model_checksum,
    )

    predicted_class = int(np.argmax(probabilities[0]))
    confidence = float(probabilities[0, predicted_class])
    print(f"Input: {preprocessing_metadata['source']}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.6f}")
    print(f"Captured ONNX node outputs: {len(layers)}")
    print(f"Reference data: {metadata_path.parent}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
