"""Export separate PyTorch reference data for padded MaxPool2D tests."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


def save_fixture(output_dir: Path, name: str, input_tensor: torch.Tensor,
                 output_tensor: torch.Tensor, kernel: int, stride: int, padding: int) -> None:
    for suffix, tensor in (("input", input_tensor), ("output", output_tensor)):
        tensor.detach().cpu().contiguous().numpy().astype("<f4").tofile(
            output_dir / f"{name}_{suffix}.bin")
    n, c, h, w = input_tensor.shape
    (output_dir / f"{name}.txt").write_text(
        f"{n} {c} {h} {w} {kernel} {stride} {padding}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "test_data" / "maxpool")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    captured = {}
    handle = model.encoder.maxpool.register_forward_hook(
        lambda _, inputs, output: captured.update(
            input=inputs[0].detach().clone(), output=output.detach().clone()
        )
    )
    try:
        with Image.open(args.image) as image:
            image, _ = resize_for_model(image.convert("RGB"), 64)
        with torch.inference_mode():
            model({"img_data": image_to_tensor(image)}, segSize=image.size[::-1])
    finally:
        handle.remove()
    save_fixture(args.output_dir, "pspnet_stem", captured["input"], captured["output"], 3, 2, 1)

    negative_input = torch.linspace(-9, -1, 2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    save_fixture(args.output_dir, "negative_padding", negative_input,
                 functional.max_pool2d(negative_input, kernel_size=3, stride=2, padding=1), 3, 2, 1)
    print(f"MaxPool2D references: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
