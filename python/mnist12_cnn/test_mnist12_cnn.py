"""Run the pretrained MNIST-12 CNN with PyTorch and export every operation.

All ``.bin`` files are raw, headerless, little-endian float32 arrays. Run
``convert_mnist12_weights.py`` once to convert the original ONNX weights into
the PyTorch state_dict consumed by this script.

Examples:
    python python/mnist12_cnn/test_mnist12_cnn.py --image digit.png
    python python/mnist12_cnn/test_mnist12_cnn.py --image digit.png --invert
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = SCRIPT_DIR / "model" / "mnist12_cnn.pth"
SOURCE_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx"
)


class MNIST12CNN(nn.Module):
    """PyTorch equivalent of the ONNX Model Zoo MNIST-12 graph."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(input_tensor)[0]

    def forward_with_intermediates(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()

        outputs["Convolution28_Output_0"] = F.conv2d(
            input_tensor,
            self.conv1.weight,
            bias=None,
            stride=1,
            padding=2,
        )
        outputs["Plus30_Output_0"] = outputs["Convolution28_Output_0"] + (
            self.conv1.bias.reshape(1, 8, 1, 1)
        )
        outputs["ReLU32_Output_0"] = F.relu(outputs["Plus30_Output_0"])
        outputs["Pooling66_Output_0"] = F.max_pool2d(
            outputs["ReLU32_Output_0"], kernel_size=2, stride=2
        )

        outputs["Convolution110_Output_0"] = F.conv2d(
            outputs["Pooling66_Output_0"],
            self.conv2.weight,
            bias=None,
            stride=1,
            padding=2,
        )
        outputs["Plus112_Output_0"] = outputs["Convolution110_Output_0"] + (
            self.conv2.bias.reshape(1, 16, 1, 1)
        )
        outputs["ReLU114_Output_0"] = F.relu(outputs["Plus112_Output_0"])
        outputs["Pooling160_Output_0"] = F.max_pool2d(
            outputs["ReLU114_Output_0"], kernel_size=3, stride=3
        )

        outputs["Pooling160_Output_0_reshape0"] = outputs[
            "Pooling160_Output_0"
        ].reshape(1, 16 * 4 * 4)
        outputs["Parameter193_reshape1"] = self.fc.weight.transpose(0, 1).contiguous()
        outputs["Times212_Output_0"] = torch.matmul(
            outputs["Pooling160_Output_0_reshape0"],
            outputs["Parameter193_reshape1"],
        )
        outputs["Plus214_Output_0"] = outputs["Times212_Output_0"] + (
            self.fc.bias.reshape(1, 10)
        )
        return outputs["Plus214_Output_0"], outputs


LAYER_DEFINITIONS = [
    {
        "node": "Convolution28",
        "op_type": "Conv",
        "output_name": "Convolution28_Output_0",
        "inputs": ["Input3", "Parameter5"],
        "attributes": {
            "kernel_shape": [5, 5],
            "strides": [1, 1],
            "pads": [2, 2, 2, 2],
            "group": 1,
            "dilations": [1, 1],
        },
    },
    {
        "node": "Plus30",
        "op_type": "Add",
        "output_name": "Plus30_Output_0",
        "inputs": ["Convolution28_Output_0", "Parameter6"],
        "attributes": {},
    },
    {
        "node": "ReLU32",
        "op_type": "Relu",
        "output_name": "ReLU32_Output_0",
        "inputs": ["Plus30_Output_0"],
        "attributes": {},
    },
    {
        "node": "Pooling66",
        "op_type": "MaxPool",
        "output_name": "Pooling66_Output_0",
        "inputs": ["ReLU32_Output_0"],
        "attributes": {"kernel_shape": [2, 2], "strides": [2, 2]},
    },
    {
        "node": "Convolution110",
        "op_type": "Conv",
        "output_name": "Convolution110_Output_0",
        "inputs": ["Pooling66_Output_0", "Parameter87"],
        "attributes": {
            "kernel_shape": [5, 5],
            "strides": [1, 1],
            "pads": [2, 2, 2, 2],
            "group": 1,
            "dilations": [1, 1],
        },
    },
    {
        "node": "Plus112",
        "op_type": "Add",
        "output_name": "Plus112_Output_0",
        "inputs": ["Convolution110_Output_0", "Parameter88"],
        "attributes": {},
    },
    {
        "node": "ReLU114",
        "op_type": "Relu",
        "output_name": "ReLU114_Output_0",
        "inputs": ["Plus112_Output_0"],
        "attributes": {},
    },
    {
        "node": "Pooling160",
        "op_type": "MaxPool",
        "output_name": "Pooling160_Output_0",
        "inputs": ["ReLU114_Output_0"],
        "attributes": {"kernel_shape": [3, 3], "strides": [3, 3]},
    },
    {
        "node": "Times212_reshape0",
        "op_type": "Reshape",
        "output_name": "Pooling160_Output_0_reshape0",
        "inputs": ["Pooling160_Output_0"],
        "attributes": {},
    },
    {
        "node": "Times212_reshape1",
        "op_type": "Reshape",
        "output_name": "Parameter193_reshape1",
        "inputs": ["fc.weight"],
        "attributes": {},
    },
    {
        "node": "Times212",
        "op_type": "MatMul",
        "output_name": "Times212_Output_0",
        "inputs": ["Pooling160_Output_0_reshape0", "Parameter193_reshape1"],
        "attributes": {},
    },
    {
        "node": "Plus214",
        "op_type": "Add",
        "output_name": "Plus214_Output_0",
        "inputs": ["Times212_Output_0", "fc.bias"],
        "attributes": {},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pretrained MNIST-12 CNN with PyTorch and export float32 "
            "reference binaries for every operation."
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
        "--weights",
        "--model",
        dest="weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Converted PyTorch state_dict created by convert_mnist12_weights.py.",
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
    with path.open("rb") as weights_file:
        for chunk in iter(lambda: weights_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(weights_path: Path) -> tuple[MNIST12CNN, Path, str]:
    resolved_path = weights_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"PyTorch weights do not exist: {resolved_path}\n"
            "Run: python python/mnist12_cnn/convert_mnist12_weights.py"
        )

    model = MNIST12CNN()
    try:
        state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(resolved_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.cpu().eval()
    return model, resolved_path, file_sha256(resolved_path)


def load_input(image_path: Path, invert: bool) -> tuple[torch.Tensor, dict[str, Any]]:
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

    input_tensor = torch.from_numpy(
        np.ascontiguousarray(image_array.reshape(1, 1, 28, 28))
    )
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
    return input_tensor, metadata


def print_model_structure(model: MNIST12CNN, input_tensor: torch.Tensor) -> None:
    with torch.inference_mode():
        _, outputs = model.forward_with_intermediates(input_tensor)

    print("Model structure:")
    print(model)
    print(
        f"  Input: Input3 shape={list(input_tensor.shape)} "
        f"dtype={input_tensor.dtype}"
    )
    for index, definition in enumerate(LAYER_DEFINITIONS):
        output = outputs[definition["output_name"]]
        print(f"  [{index:02d}] {definition['node']} ({definition['op_type']})")
        print(f"       inputs:  {', '.join(definition['inputs'])}")
        print(
            f"       output:  {definition['output_name']} "
            f"shape={list(output.shape)}"
        )
        if definition["attributes"]:
            print(f"       attributes: {definition['attributes']}")
    print("  Output: Plus214_Output_0 shape=[1, 10] dtype=torch.float32")
    print()


def run_inference(
    model: MNIST12CNN, input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    with torch.inference_mode():
        logits, outputs = model.forward_with_intermediates(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    graph_tensors: dict[str, torch.Tensor] = {
        "Input3": input_tensor,
        "Parameter5": model.conv1.weight,
        "Parameter6": model.conv1.bias,
        "Parameter87": model.conv2.weight,
        "Parameter88": model.conv2.bias,
        "fc.weight": model.fc.weight,
        "fc.bias": model.fc.bias,
    }
    layers = []
    for execution_index, definition in enumerate(LAYER_DEFINITIONS, start=1):
        output = outputs[definition["output_name"]]
        input_tensors = [
            (input_name, graph_tensors[input_name]) for input_name in definition["inputs"]
        ]
        layers.append(
            {
                "execution_index": execution_index,
                "node": definition["node"],
                "op_type": definition["op_type"],
                "output_index": 0,
                "output_name": definition["output_name"],
                "inputs": definition["inputs"],
                "attributes": definition["attributes"],
                "input_tensors": input_tensors,
                "tensor": output.detach().cpu().contiguous(),
            }
        )
        graph_tensors[definition["output_name"]] = output
    return logits.cpu(), probabilities.cpu(), layers


def tensor_layout(tensor: torch.Tensor) -> str:
    if tensor.ndim == 4:
        return "NCHW"
    if tensor.ndim == 2:
        return "NC"
    return f"{tensor.ndim}D"


def export_tensor(tensor: torch.Tensor, path: Path) -> dict[str, Any]:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    array.tofile(path)
    return {
        "file": path.name,
        "dtype": "float32-little-endian",
        "shape": list(array.shape),
        "layout": tensor_layout(tensor),
        "elements": int(array.size),
        "bytes": int(array.nbytes),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def safe_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).replace(".", "_")


def export_reference_data(
    output_dir: Path,
    model: MNIST12CNN,
    input_tensor: torch.Tensor,
    layers: list[dict[str, Any]],
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    preprocessing_metadata: dict[str, Any],
    weights_path: Path,
    weights_checksum: str,
) -> Path:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    input_metadata = export_tensor(input_tensor, resolved_output_dir / "000_input.bin")
    layer_metadata = []
    for layer in layers:
        input_metadata = []
        for input_index, (input_name, input_tensor) in enumerate(layer["input_tensors"]):
            input_filename = (
                f"{layer['execution_index']:03d}_"
                f"{safe_filename_component(str(layer['node']))}_"
                f"{safe_filename_component(str(layer['op_type']))}_input_{input_index}.bin"
            )
            input_metadata.append(
                {"name": input_name}
                | export_tensor(input_tensor, resolved_output_dir / input_filename)
            )
        filename = (
            f"{layer['execution_index']:03d}_"
            f"{safe_filename_component(str(layer['node']))}_"
            f"{safe_filename_component(str(layer['op_type']))}_output.bin"
        )
        tensor_metadata = export_tensor(
            layer["tensor"], resolved_output_dir / filename
        )
        layer_metadata.append(
            {
                key: value
                for key, value in layer.items()
                if key not in {"tensor", "input_tensors", "inputs"}
            }
            | {"inputs": input_metadata}
            | tensor_metadata
        )

    logits_metadata = export_tensor(logits, resolved_output_dir / "model_logits.bin")
    probabilities_metadata = export_tensor(
        probabilities, resolved_output_dir / "softmax_output.bin"
    )

    top_probabilities, top_indices = torch.topk(probabilities[0], k=5)
    top5 = [
        {
            "class_index": int(class_index.item()),
            "label": str(int(class_index.item())),
            "probability": float(probability.item()),
        }
        for probability, class_index in zip(top_probabilities, top_indices)
    ]
    metadata = {
        "format": "Raw headerless little-endian IEEE-754 float32",
        "model": "PyTorch MNIST12CNN converted from ONNX Model Zoo MNIST-12",
        "weights_path": str(weights_path),
        "weights_sha256": weights_checksum,
        "source_model_url": SOURCE_MODEL_URL,
        "runtime": f"PyTorch {torch.__version__}",
        "preprocessing": preprocessing_metadata,
        "prediction": top5[0],
        "top5": top5,
        "input": input_metadata,
        "layers": layer_metadata,
        "model_logits": logits_metadata,
        "softmax_output": probabilities_metadata,
        "notes": [
            "Operations preserve the original ONNX node names and execution order.",
            "Every layer input and output is exported; repeated intermediate tensors are "
            "intentionally retained per consuming layer.",
            "Conv and bias Add remain separate to preserve intermediate outputs.",
            "model_logits.bin duplicates the final Add output for convenience.",
            "Softmax is applied after the model because MNIST-12 outputs logits.",
        ],
    }

    metadata_path = resolved_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    model, weights_path, weights_checksum = load_model(args.weights)
    input_tensor, preprocessing_metadata = load_input(args.image, args.invert)
    print_model_structure(model, input_tensor)
    logits, probabilities, layers = run_inference(model, input_tensor)
    metadata_path = export_reference_data(
        output_dir=args.output_dir,
        model=model,
        input_tensor=input_tensor,
        layers=layers,
        logits=logits,
        probabilities=probabilities,
        preprocessing_metadata=preprocessing_metadata,
        weights_path=weights_path,
        weights_checksum=weights_checksum,
    )

    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_class].item())
    print(f"Input: {preprocessing_metadata['source']}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.6f}")
    print(f"Captured PyTorch operation outputs: {len(layers)}")
    print(f"Reference data: {metadata_path.parent}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
