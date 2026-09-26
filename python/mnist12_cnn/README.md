# MNIST-12 CNN Python reference workflow

This directory contains a pure PyTorch implementation of the ONNX Model Zoo
MNIST-12 CNN. It runs inference on one image and exports Vulkan reference
tensors for every original ONNX operation. Before inference, the script prints
the PyTorch model and every operation's inputs, output shape, and attributes.

The model expects a white handwritten digit on a black background. Input images
are converted to grayscale, resized to 28x28 with bilinear interpolation,
scaled to `[0, 1]`, and reshaped to NCHW `[1, 1, 28, 28]`. No mean/std
normalization is applied.

```bash
# From the repository root, activate the existing environment if needed.
source python/vkai_py/bin/activate

# Convert the original ONNX initializers to a PyTorch state_dict once.
python python/mnist12_cnn/convert_mnist12_weights.py

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

`convert_mnist12_weights.py` downloads the official 26 KB `mnist-12.onnx`
model from the ONNX Model Zoo Hugging Face mirror when necessary, verifies its
SHA-256 checksum, and creates `model/mnist12_cnn.pth`. This conversion step
requires `onnx`; install `requirements-convert.txt` if it is not available.

`test_mnist12_cnn.py` does not import ONNX or use an ONNX runtime. Its model,
inference, softmax, and intermediate tensors are implemented with PyTorch.
It also exports `model/mnist12_weights.bin` in VKAI FlatBuffers format for the
C++ Vulkan MNIST-12 task. Override that destination with `--vkai-weights-output`.

By default, generated reference files are written to `test_data/`:

- `000_input.bin`: preprocessed NCHW model input.
- `NNN_<node>_<op>_input_<index>.bin`: every input consumed by each operation,
  including weights and biases.
- `NNN_<node>_<op>_output.bin`: every corresponding PyTorch operation output.
- `model_logits.bin`: final 10-class logits before softmax.
- `softmax_output.bin`: final probabilities.
- `metadata.json`: shapes, layouts, ONNX attributes, preprocessing, and top-5
  predictions.

Every `.bin` is a raw, headerless, little-endian IEEE-754 float32 array. Conv
and bias Add remain separate so the 12 exported outputs retain the original
ONNX names and order: two `Conv -> Add -> Relu -> MaxPool` groups followed by
two `Reshape` operations, `MatMul`, and `Add`.

Use `--output-dir` to write the generated files somewhere else, or `--weights`
to load a different compatible PyTorch state_dict:

```bash
python python/mnist12_cnn/test_mnist12_cnn.py \
    --image python/mnist12_cnn/test_digit_7.png \
    --output-dir /tmp/mnist12-reference
```
