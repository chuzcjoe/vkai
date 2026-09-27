"""Export pretrained PSPNet weights and an end-to-end Vulkan test fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import flatbuffers
import numpy as np
import torch
from PIL import Image

from infer import image_to_tensor, load_model, resize_for_model, select_device


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from vkai.fbs import Model, Tensor


def export_weights(model: torch.nn.Module, destination: Path) -> None:
    builder = flatbuffers.Builder(256 * 1024 * 1024)
    tensor_offsets = []
    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        tensor_name = builder.CreateString(name)
        shape = builder.CreateNumpyVector(np.asarray(array.shape, dtype="<u4"))
        data = builder.CreateNumpyVector(array.reshape(-1))
        Tensor.TensorStart(builder)
        Tensor.TensorAddName(builder, tensor_name)
        Tensor.TensorAddShape(builder, shape)
        Tensor.TensorAddData(builder, data)
        tensor_offsets.append(Tensor.TensorEnd(builder))
    model_name = builder.CreateString("pspnet_resnet50dilated_ppm")
    Model.ModelStartWeightsVector(builder, len(tensor_offsets))
    for offset in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(offset)
    weights = builder.EndVector()
    Model.ModelStart(builder)
    Model.ModelAddName(builder, model_name)
    Model.ModelAddWeights(builder, weights)
    builder.Finish(Model.ModelEnd(builder), file_identifier=b"VKAI")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(builder.Output())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=SCRIPT_DIR / "models")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "test_data" / "end_to_end")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "model" / "pspnet_weights.bin")
    parser.add_argument("--max-size", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir.resolve(), select_device("cpu"))
    with Image.open(args.image) as image:
        model_image, _ = resize_for_model(image.convert("RGB"), args.max_size)
    input_tensor = image_to_tensor(model_image)
    with torch.inference_mode():
        output_tensor = model({"img_data": input_tensor}, segSize=model_image.size[::-1])
    input_tensor.numpy().astype("<f4", copy=False).tofile(args.output_dir / "input.bin")
    output_tensor.numpy().astype("<f4", copy=False).tofile(args.output_dir / "probabilities.bin")
    _, channels, height, width = output_tensor.shape
    (args.output_dir / "shape.txt").write_text(f"1 {channels} {height} {width}\n")
    export_weights(model, args.weights)
    print(f"PSPNet input/output: {args.output_dir.resolve()}")
    print(f"PSPNet Vulkan weights: {args.weights.resolve()}")


if __name__ == "__main__":
    main()
