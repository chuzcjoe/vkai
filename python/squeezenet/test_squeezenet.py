"""Run SqueezeNet 1.1 inference and export every layer output.

All ``.bin`` files are raw, headerless, little-endian float32 arrays. Tensor
shapes, module names, preprocessing details, and predictions are recorded in
``metadata.json``.

Example:
    python python/squeezenet/test_squeezenet.py --image cat.jpg
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained SqueezeNet 1.1 on one image and export float32 "
            "reference binaries for every executed layer."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image. It is converted to RGB and preprocessed by torchvision.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "test_data",
        help="Directory for exported .bin files and metadata.json.",
    )
    return parser.parse_args()


def load_input(
    image_path: Path, weights: SqueezeNet1_1_Weights
) -> tuple[torch.Tensor, dict[str, Any]]:
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {resolved_path}")

    preprocessing = weights.transforms()
    with Image.open(resolved_path) as opened_image:
        original_mode = opened_image.mode
        original_size = list(opened_image.size)
        rgb_image = opened_image.convert("RGB")
        input_tensor = preprocessing(rgb_image).unsqueeze(0)

    metadata = {
        "source": str(resolved_path),
        "original_mode": original_mode,
        "original_size_wh": original_size,
        "resize_size": list(preprocessing.resize_size),
        "crop_size": list(preprocessing.crop_size),
        "interpolation": str(preprocessing.interpolation),
        "antialias": bool(preprocessing.antialias),
        "mean": list(preprocessing.mean),
        "std": list(preprocessing.std),
        "steps": [
            "convert to RGB",
            "bilinear resize",
            "center crop",
            "convert pixels to float32 in [0, 1]",
            "normalize each RGB channel with mean and std",
            "add the N dimension to produce NCHW input",
        ],
    }
    return input_tensor, metadata


def should_capture(module: nn.Module) -> bool:
    """Capture leaf operations and Fire outputs after their channel concat."""
    is_leaf = not any(module.children())
    is_fire_block = module.__class__.__name__ == "Fire"
    return is_leaf or is_fire_block


def run_inference(
    model: nn.Module, input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    captured_layers: list[dict[str, Any]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(module_name: str):
        def capture_output(module: nn.Module, _inputs: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Unsupported non-tensor output from {module_name}: {type(output)}"
                )

            # Clone inside the hook. Several SqueezeNet ReLUs are in-place and
            # would otherwise mutate an earlier module's captured output.
            captured_layers.append(
                {
                    "module": module_name,
                    "type": module.__class__.__name__,
                    "tensor": output.detach().cpu().clone().contiguous(),
                }
            )

        return capture_output

    for module_name, module in model.named_modules():
        if module_name and should_capture(module):
            handles.append(module.register_forward_hook(make_hook(module_name)))

    try:
        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
    finally:
        for handle in handles:
            handle.remove()

    return logits.cpu(), probabilities.cpu(), captured_layers


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
    input_tensor: torch.Tensor,
    layers: list[dict[str, Any]],
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    preprocessing_metadata: dict[str, Any],
    weights: SqueezeNet1_1_Weights,
) -> Path:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    input_metadata = export_tensor(
        input_tensor, resolved_output_dir / "000_input.bin"
    )

    layer_metadata = []
    for index, layer in enumerate(layers, start=1):
        module_name = str(layer["module"])
        module_type = str(layer["type"])
        filename = (
            f"{index:03d}_{safe_filename_component(module_name)}_"
            f"{safe_filename_component(module_type)}_output.bin"
        )
        tensor_metadata = export_tensor(
            layer["tensor"], resolved_output_dir / filename
        )
        layer_metadata.append(
            {
                "execution_index": index,
                "module": module_name,
                "type": module_type,
                **tensor_metadata,
            }
        )

    logits_metadata = export_tensor(
        logits, resolved_output_dir / "model_logits.bin"
    )
    probabilities_metadata = export_tensor(
        probabilities, resolved_output_dir / "softmax_output.bin"
    )

    top_probabilities, top_indices = torch.topk(probabilities[0], k=5)
    categories = weights.meta["categories"]
    top5 = [
        {
            "class_index": int(class_index.item()),
            "label": categories[int(class_index.item())],
            "probability": float(probability.item()),
        }
        for probability, class_index in zip(top_probabilities, top_indices)
    ]

    metadata = {
        "format": "Raw headerless little-endian IEEE-754 float32",
        "model": "torchvision.models.squeezenet1_1",
        "weights": str(weights),
        "weights_url": weights.url,
        "device": "cpu",
        "preprocessing": preprocessing_metadata,
        "prediction": top5[0],
        "top5": top5,
        "input": input_metadata,
        "layers": layer_metadata,
        "model_logits": logits_metadata,
        "softmax_output": probabilities_metadata,
        "notes": [
            "Layers are ordered by actual forward execution.",
            "Leaf nn.Module outputs and complete Fire block outputs are exported.",
            "Fire outputs include the functional torch.cat result.",
            "model_logits.bin is the flattened model output before softmax.",
        ],
    }

    metadata_path = resolved_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    weights = SqueezeNet1_1_Weights.DEFAULT
    model = squeezenet1_1(weights=weights).cpu().eval()
    input_tensor, preprocessing_metadata = load_input(args.image, weights)
    logits, probabilities, layers = run_inference(model, input_tensor)
    metadata_path = export_reference_data(
        output_dir=args.output_dir,
        input_tensor=input_tensor,
        layers=layers,
        logits=logits,
        probabilities=probabilities,
        preprocessing_metadata=preprocessing_metadata,
        weights=weights,
    )

    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    predicted_label = weights.meta["categories"][predicted_index]
    confidence = float(probabilities[0, predicted_index].item())

    print(f"Input: {preprocessing_metadata['source']}")
    print(f"Predicted class: {predicted_index} ({predicted_label})")
    print(f"Confidence: {confidence:.6f}")
    print(f"Captured layer outputs: {len(layers)}")
    print(f"Reference data: {metadata_path.parent}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
