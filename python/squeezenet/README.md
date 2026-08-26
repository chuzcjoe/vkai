# SqueezeNet 1.1 Python reference workflow

This directory runs the pretrained TorchVision SqueezeNet 1.1 model on one
image and exports Vulkan reference tensors for every executed layer.

The script uses the preprocessing bundled with the official pretrained
weights: RGB conversion, bilinear resize to 256, center crop to 224, conversion
to `[0, 1]` float32, and ImageNet mean/std normalization.

```bash
# From the repository root, activate the existing environment if needed.
source python/vkai_py/bin/activate

# Run inference. The pretrained weights are downloaded automatically once and
# then reused from PyTorch's local cache.
python python/squeezenet/test_squeezenet.py --image /path/to/image.jpg
```

By default, generated files are written to `test_data/`:

- `000_input.bin`: preprocessed NCHW model input.
- `NNN_<module>_<type>_output.bin`: layer outputs in forward execution order.
- `model_logits.bin`: final 1000-class logits before softmax.
- `softmax_output.bin`: final probabilities.
- `metadata.json`: shapes, layouts, module names, preprocessing, and top-5
  predictions.

Every `.bin` is a raw, headerless, little-endian IEEE-754 float32 array. Leaf
module outputs are captured along with each complete Fire block output. Saving
the Fire output is important because it includes the block's functional
channel concatenation.

Use `--output-dir` to write the generated files somewhere else:

```bash
python python/squeezenet/test_squeezenet.py \
    --image /path/to/image.jpg \
    --output-dir /tmp/squeezenet-reference
```
