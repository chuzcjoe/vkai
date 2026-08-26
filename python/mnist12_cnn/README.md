# MNIST-12 CNN Python reference workflow

This directory runs the pretrained ONNX Model Zoo MNIST-12 CNN on one image
and exports Vulkan reference tensors for every ONNX node output.

The model expects a white handwritten digit on a black background. Input images
are converted to grayscale, resized to 28x28 with bilinear interpolation,
scaled to `[0, 1]`, and reshaped to NCHW `[1, 1, 28, 28]`. No mean/std
normalization is applied.

```bash
# From the repository root, activate the existing environment if needed.
source python/vkai_py/bin/activate

# White digit on a black background.
python python/mnist12_cnn/test_mnist12_cnn.py --image /path/to/digit.png

# Run the bundled white handwritten 7 test image.
python python/mnist12_cnn/test_mnist12_cnn.py \
    --image python/mnist12_cnn/test_digit_7.png

# Dark digit on a light background.
python python/mnist12_cnn/test_mnist12_cnn.py \
    --image /path/to/digit.png \
    --invert
```

The official 26 KB `mnist-12.onnx` model is downloaded automatically from the
ONNX Model Zoo Hugging Face mirror on first use. Its SHA-256 checksum is
verified before it is cached under `model/`.

By default, generated reference files are written to `test_data/`:

- `000_input.bin`: preprocessed NCHW model input.
- `NNN_<node>_<op>_output.bin`: every ONNX node output in graph order.
- `model_logits.bin`: final 10-class logits before softmax.
- `softmax_output.bin`: final probabilities.
- `metadata.json`: shapes, layouts, ONNX attributes, preprocessing, and top-5
  predictions.

Every `.bin` is a raw, headerless, little-endian IEEE-754 float32 array. The
model contains 12 nodes: two `Conv -> Add -> Relu -> MaxPool` groups followed
by two `Reshape` nodes, `MatMul`, and `Add`.

Use `--output-dir` to write the generated files somewhere else, or `--model`
to use an existing compatible ONNX model:

```bash
python python/mnist12_cnn/test_mnist12_cnn.py \
    --image /path/to/digit.png \
    --output-dir /tmp/mnist12-reference
```
