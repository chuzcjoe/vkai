"""Export PSPNet channel-softmax PyTorch reference tensors."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


def save_fixture(directory: Path, name: str, input_tensor: torch.Tensor,
                 output_tensor: torch.Tensor) -> None:
    for suffix, tensor in (("input", input_tensor), ("output", output_tensor)):
        tensor.detach().cpu().contiguous().numpy().astype("<f4").tofile(
            directory / f"{name}_{suffix}.bin"
        )
    n, c, h, w = input_tensor.shape
    (directory / f"{name}.txt").write_text(f"{n} {c} {h} {w}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "test_data" / "softmax")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    logits = {}
    handle = model.decoder.conv_last[-1].register_forward_hook(
        lambda _, __, output: logits.update(value=output.detach().clone())
    )
    try:
        with Image.open(args.image) as image:
            image, _ = resize_for_model(image.convert("RGB"), 64)
        with torch.inference_mode():
            probabilities = model({"img_data": image_to_tensor(image)}, segSize=image.size[::-1])
    finally:
        handle.remove()
    final_logits = functional.interpolate(logits["value"], size=probabilities.shape[-2:],
                                          mode="bilinear", align_corners=False)
    save_fixture(args.output_dir, "pspnet_final", final_logits, probabilities)

    batched_logits = torch.linspace(-90, 90, 2 * 5 * 3 * 7).reshape(2, 5, 3, 7)
    save_fixture(args.output_dir, "batched_extreme", batched_logits,
                 functional.softmax(batched_logits, dim=1))
    print(f"Softmax references: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
