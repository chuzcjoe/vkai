"""Export isolated PyTorch references for PSPNet channel concatenation."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


def save_fixture(directory: Path, name: str, inputs: list[torch.Tensor], output: torch.Tensor) -> None:
    for index, tensor in enumerate(inputs):
        tensor.detach().cpu().contiguous().numpy().astype("<f4").tofile(
            directory / f"{name}_input_{index}.bin"
        )
    output.detach().cpu().contiguous().numpy().astype("<f4").tofile(
        directory / f"{name}_output.bin"
    )
    batch, _, height, width = output.shape
    channels = " ".join(str(tensor.shape[1]) for tensor in inputs)
    (directory / f"{name}.txt").write_text(
        f"{batch} {height} {width} {len(inputs)} {channels}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "test_data" / "channel_concat")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    captured = {"branches": {}}
    handles = []
    for index, branch in enumerate(model.decoder.ppm):
        handles.append(branch.register_forward_hook(
            lambda _, __, output, index=index: captured["branches"].update(
                {index: output.detach().clone()}
            )
        ))
    handles.append(model.decoder.conv_last[0].register_forward_pre_hook(
        lambda _, inputs: captured.update(output=inputs[0].detach().clone())
    ))
    try:
        with Image.open(args.image) as image:
            image, _ = resize_for_model(image.convert("RGB"), 64)
        with torch.inference_mode():
            model({"img_data": image_to_tensor(image)}, segSize=image.size[::-1])
    finally:
        for handle in handles:
            handle.remove()

    output = captured["output"]
    height, width = output.shape[-2:]
    original = output[:, :2048].clone()
    branch_inputs = [
        functional.interpolate(captured["branches"][index], size=(height, width),
                               mode="bilinear", align_corners=False)
        for index in range(4)
    ]
    save_fixture(args.output_dir, "pspnet_ppm", [original, *branch_inputs], output)

    first = torch.linspace(-9, 2, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5)
    second = torch.linspace(3, 9, 2 * 3 * 3 * 5).reshape(2, 3, 3, 5)
    save_fixture(args.output_dir, "batched_two_inputs", [first, second], torch.cat([first, second], 1))
    print(f"ChannelConcat references: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
