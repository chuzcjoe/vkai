"""Export separate PyTorch references for PSPNet bilinear interpolation."""

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
    n, c, input_height, input_width = input_tensor.shape
    _, _, output_height, output_width = output_tensor.shape
    (directory / f"{name}.txt").write_text(
        f"{n} {c} {input_height} {input_width} {output_height} {output_width}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "test_data" / "bilinear_resize")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    pooled = {}
    logits = {}
    handles = []
    for index, pool_branch in enumerate(model.decoder.ppm):
        handles.append(pool_branch[0].register_forward_hook(
            lambda _, __, output, index=index: pooled.update({index: output.detach().clone()})
        ))
    handles.append(model.decoder.conv_last[-1].register_forward_hook(
        lambda _, __, output: logits.update(value=output.detach().clone())
    ))
    try:
        with Image.open(args.image) as image:
            image, _ = resize_for_model(image.convert("RGB"), 64)
        with torch.inference_mode():
            model({"img_data": image_to_tensor(image)}, segSize=image.size[::-1])
    finally:
        for handle in handles:
            handle.remove()

    for index, input_tensor in pooled.items():
        output_tensor = functional.interpolate(input_tensor, size=(8, 8), mode="bilinear",
                                               align_corners=False)
        save_fixture(args.output_dir, f"ppm_scale_{input_tensor.shape[-1]}", input_tensor,
                     output_tensor)
    save_fixture(args.output_dir, "final_logits", logits["value"],
                 functional.interpolate(logits["value"], size=(64, 64), mode="bilinear",
                                        align_corners=False))
    batched_input = torch.linspace(-7, 13, 2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    save_fixture(args.output_dir, "batched_non_square", batched_input,
                 functional.interpolate(batched_input, size=(11, 4), mode="bilinear",
                                        align_corners=False))
    print(f"BilinearResize2D references: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
