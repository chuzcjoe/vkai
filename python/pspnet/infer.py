"""Run pretrained PSPNet semantic segmentation on one RGB image.

The model is MIT CSAIL's ResNet-50-dilated + PPM_deepsup PSPNet trained on
ADE20K. On its first run this script downloads the official encoder and decoder
checkpoints into ``python/pspnet/models``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "ade20k-resnet50dilated-ppm_deepsup"
# The upstream demo publishes these legacy checkpoints over HTTP.
MODEL_BASE_URL = "http://sceneparsing.csail.mit.edu/model/pytorch"
ENCODER_FILENAME = "encoder_epoch_20.pth"
DECODER_FILENAME = "decoder_epoch_20.pth"
NUM_CLASSES = 150
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment one image with pretrained ADE20K PSPNet."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input RGB image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Directory for mask.png, overlay.png, and metadata.json.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=SCRIPT_DIR / "models",
        help="Directory used to cache official PSPNet checkpoints.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Resize the input so its longest side is at most this value (default: 512).",
    )
    args = parser.parse_args()
    if args.max_size <= 0:
        parser.error("--max-size must be positive")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def import_mit_semseg() -> tuple[Any, Any]:
    try:
        from mit_semseg.models import ModelBuilder, SegmentationModule
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'mit_semseg'. Install the official implementation with:\n"
            "  python -m pip install "
            "git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master"
        ) from error
    return ModelBuilder, SegmentationModule


def download_checkpoint(model_dir: Path, filename: str) -> Path:
    destination = model_dir / filename
    if destination.is_file():
        return destination

    model_dir.mkdir(parents=True, exist_ok=True)
    source = f"{MODEL_BASE_URL}/{MODEL_NAME}/{filename}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(source) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download pretrained PSPNet checkpoint: {source}") from error
    return destination


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    model_builder, segmentation_module = import_mit_semseg()
    encoder_path = download_checkpoint(model_dir, ENCODER_FILENAME)
    decoder_path = download_checkpoint(model_dir, DECODER_FILENAME)
    encoder = model_builder.build_encoder(
        arch="resnet50dilated", fc_dim=2048, weights=str(encoder_path)
    )
    decoder = model_builder.build_decoder(
        arch="ppm_deepsup",
        fc_dim=2048,
        num_class=NUM_CLASSES,
        weights=str(decoder_path),
        use_softmax=True,
    )
    model = segmentation_module(encoder, decoder, nn.NLLLoss())
    return model.to(device).eval()


def resize_for_model(image: Image.Image, max_size: int) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    scale = min(1.0, max_size / max(width, height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    padded_width = (resized_width + 7) // 8 * 8
    padded_height = (resized_height + 7) // 8 * 8
    if (padded_width, padded_height) == resized.size:
        return resized, resized.size
    canvas = Image.new("RGB", (padded_width, padded_height))
    canvas.paste(resized, (0, 0))
    return canvas, resized.size


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).unsqueeze(0)


def ade20k_palette() -> list[int]:
    palette = []
    for label in range(256):
        value = label
        red = green = blue = 0
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette.extend((red, green, blue))
    return palette


def save_results(
    original: Image.Image, labels: np.ndarray, output_dir: Path, device: torch.device
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_image = Image.fromarray(labels.astype(np.uint8), mode="P")
    label_image.putpalette(ade20k_palette())
    label_image.save(output_dir / "mask.png")

    colored = label_image.convert("RGB")
    Image.blend(original, colored, alpha=0.5).save(output_dir / "overlay.png")

    labels_present, counts = np.unique(labels, return_counts=True)
    ranked = sorted(zip(labels_present.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)
    metadata = {
        "model": "PSPNet ResNet-50-dilated + PPM_deepsup trained on ADE20K",
        "num_classes": NUM_CLASSES,
        "device": str(device),
        "input_size_wh": list(original.size),
        "predicted_classes": [
            {
                "ade20k_class_id": class_id + 1,
                "pixel_count": pixel_count,
                "coverage_percent": round(100.0 * pixel_count / labels.size, 4),
            }
            for class_id, pixel_count in ranked
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    device = select_device(args.device)
    with Image.open(image_path) as opened_image:
        original = opened_image.convert("RGB")
    model_image, resized_size = resize_for_model(original, args.max_size)
    input_tensor = image_to_tensor(model_image).to(device)
    model = load_model(args.model_dir.expanduser().resolve(), device)

    with torch.inference_mode():
        probabilities = model({"img_data": input_tensor}, segSize=model_image.size[::-1])
        labels = probabilities.argmax(dim=1)[0, : resized_size[1], : resized_size[0]]
    label_image = Image.fromarray(labels.cpu().numpy().astype(np.uint8), mode="L")
    labels_at_input_resolution = np.asarray(
        label_image.resize(original.size, Image.Resampling.NEAREST), dtype=np.uint8
    )
    output_dir = args.output_dir.expanduser().resolve()
    save_results(original, labels_at_input_resolution, output_dir, device)
    print(f"Segmentation mask: {output_dir / 'mask.png'}")
    print(f"Segmentation overlay: {output_dir / 'overlay.png'}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
