"""Export isolated PSPNet AdaptiveAvgPool2D PyTorch references."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().numpy().astype("<f4").tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "test_data" / "adaptive_avg_pool")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    captured = {}
    handles = []
    for index, pool_branch in enumerate(model.decoder.ppm):
        def capture(_, inputs, output, index=index):
            captured[index] = (inputs[0].detach().clone(), output.detach().clone())
        handles.append(pool_branch[0].register_forward_hook(capture))
    try:
        with Image.open(args.image) as image:
            image, _ = resize_for_model(image.convert("RGB"), 64)
        with torch.inference_mode():
            model({"img_data": image_to_tensor(image)}, segSize=image.size[::-1])
    finally:
        for handle in handles:
            handle.remove()

    for index, (input_tensor, output_tensor) in captured.items():
        name = f"pspnet_scale_{output_tensor.shape[-1]}"
        save_tensor(input_tensor, args.output_dir / f"{name}_input.bin")
        save_tensor(output_tensor, args.output_dir / f"{name}_output.bin")
        n, c, input_height, input_width = input_tensor.shape
        _, _, output_height, output_width = output_tensor.shape
        (args.output_dir / f"{name}.txt").write_text(
            f"{n} {c} {input_height} {input_width} {output_height} {output_width}\n"
        )

    batched_input = torch.linspace(-7, 11, 2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    batched_output = functional.adaptive_avg_pool2d(batched_input, (3, 4))
    save_tensor(batched_input, args.output_dir / "batched_input.bin")
    save_tensor(batched_output, args.output_dir / "batched_output.bin")
    (args.output_dir / "batched.txt").write_text("2 3 5 7 3 4\n")
    print(f"AdaptiveAvgPool2D references: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
