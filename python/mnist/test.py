"""Run one MNIST inference and export reference tensors for C++ unit tests.

Examples:
    # Use item 0 from the MNIST test set.
    python python/mnist/test.py --index 0

    # Use a custom image (expected to be a light digit on a dark background).
    python python/mnist/test.py --image digit.png

    # Invert a dark-on-light custom image before inference.
    python python/mnist/test.py --image digit.png --invert

Every ``.bin`` file is a raw, headerless, little-endian float32 array. Tensor
shapes and inference details are written to ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import datasets, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTNet(nn.Module):
    """Network architecture used by train.py."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one MNIST inference as float32 reference binaries."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        type=Path,
        help="Custom input image. It is converted to grayscale and resized to 28x28.",
    )
    source.add_argument(
        "--index",
        type=int,
        default=0,
        help="MNIST test-set index to use when --image is not supplied (default: 0).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert a custom image, useful for a dark digit on a light background.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR / "mnist_model.pth",
        help="Path to the trained PyTorch state_dict.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="MNIST dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "test_data",
        help="Directory for exported .bin files and metadata.json.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the MNIST test set if it is missing.",
    )
    return parser.parse_args()


def preprocessing_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def load_input(args: argparse.Namespace) -> tuple[torch.Tensor, Optional[int], str]:
    transform = preprocessing_transform()

    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        with Image.open(image_path) as opened_image:
            image = opened_image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
            if args.invert:
                image = ImageOps.invert(image)
            input_tensor = transform(image)

        return input_tensor.unsqueeze(0), None, str(image_path)

    if args.index < 0:
        raise ValueError("--index must be non-negative")

    dataset = datasets.MNIST(
        root=str(args.data_dir.expanduser()),
        train=False,
        transform=transform,
        download=args.download,
    )
    if args.index >= len(dataset):
        raise IndexError(
            f"MNIST index {args.index} is out of range; valid range is 0..{len(dataset) - 1}"
        )

    input_tensor, label = dataset[args.index]
    return input_tensor.unsqueeze(0), int(label), f"MNIST test index {args.index}"


def load_model(model_path: Path) -> MNISTNet:
    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {resolved_path}")

    model = MNISTNet()
    try:
        state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        state_dict = torch.load(resolved_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_inference(
    model: MNISTNet, input_tensor: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Run each operation explicitly so every Vulkan-visible output is captured."""
    with torch.inference_mode():
        flattened_input = input_tensor.reshape(1, 28 * 28)
        fc1_output = model.fc1(flattened_input)
        relu1_output = model.relu(fc1_output)
        fc2_output = model.fc2(relu1_output)
        softmax_output = torch.softmax(fc2_output, dim=1)

    return {
        "input": input_tensor,
        "fc1_output": fc1_output,
        "relu1_output": relu1_output,
        "fc2_output": fc2_output,
        "softmax_output": softmax_output,
    }


def export_tensor(tensor: torch.Tensor, path: Path) -> dict[str, object]:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    array.tofile(path)
    return {
        "file": path.name,
        "dtype": "float32-little-endian",
        "shape": list(array.shape),
        "elements": int(array.size),
        "bytes": int(array.nbytes),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def export_reference_data(
    tensors: dict[str, torch.Tensor],
    output_dir: Path,
    source: str,
    label: Optional[int],
    model_path: Path,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_metadata = {}
    for name, tensor in tensors.items():
        tensor_metadata[name] = export_tensor(tensor, output_dir / f"{name}.bin")

    probabilities = tensors["softmax_output"][0]
    predicted_class = int(torch.argmax(probabilities).item())
    metadata = {
        "format": "Raw headerless little-endian IEEE-754 float32",
        "model": str(model_path.expanduser().resolve()),
        "source": source,
        "expected_label": label,
        "predicted_class": predicted_class,
        "confidence": float(probabilities[predicted_class].item()),
        "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
        "layout": {
            "input": "NCHW",
            "fc1_output": "NC",
            "relu1_output": "NC",
            "fc2_output": "NC",
            "softmax_output": "NC",
        },
        "tensors": tensor_metadata,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    model = load_model(args.model)
    input_tensor, label, source = load_input(args)
    tensors = run_inference(model, input_tensor)
    metadata_path = export_reference_data(
        tensors=tensors,
        output_dir=args.output_dir,
        source=source,
        label=label,
        model_path=args.model,
    )

    predicted_class = int(torch.argmax(tensors["softmax_output"], dim=1).item())
    confidence = float(tensors["softmax_output"][0, predicted_class].item())

    print(f"Input: {source}")
    if label is not None:
        print(f"Expected label: {label}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.6f}")
    print(f"Reference data: {metadata_path.parent}")
    for name, tensor in tensors.items():
        print(f"  {name}.bin: shape={list(tensor.shape)}, elements={tensor.numel()}")
    print(f"  {metadata_path.name}")


if __name__ == "__main__":
    main()
