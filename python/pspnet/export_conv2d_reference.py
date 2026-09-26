"""Export pretrained PSPNet Conv2D inputs, parameters, and outputs for C++ tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real pretrained PSPNet Conv2D reference tensors."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input scene image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "test_data",
        help="Directory for raw float32 tensor files and metadata.json.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=SCRIPT_DIR / "models",
        help="Directory containing PSPNet checkpoints.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=64,
        help="Maximum image side used for the compact test reference (default: 64).",
    )
    return parser.parse_args()


def export_tensor(tensor: torch.Tensor, path: Path) -> dict[str, Any]:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    array.tofile(path)
    return {"file": path.name, "shape": list(array.shape), "elements": int(array.size)}


def capture_conv(module: torch.nn.Conv2d, name: str, captured: dict[str, Any]) -> Any:
    def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured[name] = {"input": inputs[0].detach(), "output": output.detach()}

    return module.register_forward_hook(hook)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if args.max_size <= 0:
        raise ValueError("--max-size must be positive")

    device = select_device("cpu")
    model = load_model(args.model_dir.expanduser().resolve(), device)
    selected_layers = {
        "dilated_conv": model.encoder.layer3[1].conv2,
        "classifier_conv": model.decoder.conv_last[-1],
    }
    captured: dict[str, Any] = {}
    hooks = [capture_conv(layer, name, captured) for name, layer in selected_layers.items()]
    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        model_image, _ = resize_for_model(image, args.max_size)
        with torch.inference_mode():
            model({"img_data": image_to_tensor(model_image).to(device)}, segSize=model_image.size[::-1])
    finally:
        for hook in hooks:
            hook.remove()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"layers": {}}
    for name, layer in selected_layers.items():
        layer_data = captured[name]
        layer_metadata = {
            "input": export_tensor(layer_data["input"], output_dir / f"{name}_input.bin"),
            "weight": export_tensor(layer.weight, output_dir / f"{name}_weight.bin"),
            "output": export_tensor(layer_data["output"], output_dir / f"{name}_output.bin"),
            "kernel_size": list(layer.kernel_size),
            "stride": list(layer.stride),
            "padding": list(layer.padding),
            "dilation": list(layer.dilation),
            "groups": layer.groups,
        }
        if layer.bias is not None:
            layer_metadata["bias"] = export_tensor(layer.bias, output_dir / f"{name}_bias.bin")
        metadata["layers"][name] = layer_metadata
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PSPNet Conv2D reference data: {output_dir}")


if __name__ == "__main__":
    main()
