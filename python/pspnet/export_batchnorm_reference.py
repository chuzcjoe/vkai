"""Export isolated BN fixtures without modifying existing Conv2D/Add references."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "test_data" / "batchnorm")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}

    def save(name, layer, x, y):
        n, c, h, w = x.shape
        tensors = dict(input=x, output=y, mean=layer.running_mean, variance=layer.running_var,
                       weight=layer.weight if layer.weight is not None else torch.empty(0),
                       bias=layer.bias if layer.bias is not None else torch.empty(0))
        for key, tensor in tensors.items():
            tensor.detach().cpu().contiguous().numpy().astype("<f4").tofile(
                args.output_dir / f"{name}_{key}.bin")
        (args.output_dir / f"{name}.txt").write_text(f"{n} {c} {h} {w} {layer.eps}\n")
        metadata[name] = dict(shape=list(x.shape), eps=layer.eps, affine=layer.affine)

    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    model.eval()
    handles = []
    for name, layer in {"pspnet_bn2": model.encoder.layer3[1].bn2,
                        "pspnet_bn3": model.encoder.layer3[1].bn3}.items():
        def capture(module, inputs, output, name=name):
            # Clone before downstream in-place ReLU/residual operations can mutate storage.
            save(name, module, inputs[0].detach().clone(), output.detach().clone())
        handles.append(layer.register_forward_hook(capture))
    try:
        with Image.open(args.image) as image:
            image, _ = resize_for_model(image.convert("RGB"), 64)
        with torch.inference_mode():
            model({"img_data": image_to_tensor(image)}, segSize=image.size[::-1])
    finally:
        for handle in handles:
            handle.remove()

    for name, affine in [("batched", True), ("no_affine", False)]:
        bn = torch.nn.BatchNorm2d(3, eps=0.003, affine=affine).eval()
        with torch.no_grad():
            bn.running_mean.copy_(torch.tensor([-2.0, 0.5, 3.0]))
            bn.running_var.copy_(torch.tensor([0.0, 0.2, 4.0]))
            if affine:
                bn.weight.copy_(torch.tensor([-1.5, 0.0, 2.0]))
                bn.bias.copy_(torch.tensor([0.25, -0.5, 1.0]))
            x = torch.linspace(-5, 5, 2 * 3 * 7 * 13).reshape(2, 3, 7, 13)
            save(name, bn, x, bn(x))
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"BatchNorm references: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
