# PSPNet semantic segmentation

`infer.py` runs MIT CSAIL's pretrained PSPNet scene parser: a ResNet-50-dilated
encoder with a Pyramid Pooling Module decoder, trained for 150 ADE20K semantic
classes. It automatically downloads the official encoder and decoder checkpoints
on first use.

```bash
source python/vkai_py/bin/activate
python -m pip install -r python/pspnet/requirements.txt

python python/pspnet/infer.py --image /path/to/photo.jpg
```

Outputs default to `python/pspnet/output/`:

- `mask.png`: 0-based ADE20K class map, rendered with a deterministic palette.
- `overlay.png`: original image blended with the predicted class colors.
- `metadata.json`: model details and class-id pixel coverage.

Use `--max-size` to trade accuracy for speed. The default is 512 pixels on the
longest side, and CPU inference may take a while. `--device auto` selects CUDA
when it is available and otherwise uses CPU.

The implementation and pretrained ResNet-50 + PPM model are from MIT CSAIL's
[semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
project, licensed under BSD-3-Clause.

For C++ Conv2D regression tests, export compact real PSPNet tensors with:

```bash
python python/pspnet/export_conv2d_reference.py --image /path/to/photo.jpg
```

This exports a dilated ResNet bottleneck convolution, the bias-bearing PSPNet
classifier convolution, and a ResNet residual Add to `test_data/`. The generated
binary tensors are ignored by Git.
