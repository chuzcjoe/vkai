# VKAI

A minimal C++ inference demo using FlatBuffers. The build converts fake model
weights from `assets/weights.json` into `build/weights.bin`; the executable maps
that file with `mmap`, accesses its float32 tensors in place, and runs a small
linear layer. The Protobuf dependency and schema remain in the repository but
are not currently built or used.

The MNIST training, generated artifacts, and C++ reference-data workflow live
in [`python/mnist`](python/mnist/README.md).

The pretrained MNIST-12 CNN per-node reference export workflow lives in
[`python/mnist12_cnn`](python/mnist12_cnn/README.md).

The pretrained SqueezeNet 1.1 per-layer reference export workflow lives in
[`python/squeezenet`](python/squeezenet/README.md).

## Model artifact and reference-data generation

Run these commands from the repository root.  Each model should have one row
here as it is added: the first command produces the inference artifact and the
second produces the reference tensors consumed by its C++ tests.

| Model | Python dependencies | Generate inference artifact | Generate C++ test reference data | Generated files |
| --- | --- | --- | --- | --- |
| MNIST (784 → 128 → 10) | `source python/vkai_py/bin/activate`<br>`pip install -r python/mnist/requirements.txt` | `python python/mnist/train_mnist.py --skip-onnx` | `python python/mnist/test_mnist.py --index 0` | `python/mnist/mnist_weights.bin` (FlatBuffer weights), `python/mnist/mnist_model.pth`, and `python/mnist/test_data/*.bin` plus `metadata.json` |
| _Future model_ | _Add its requirements or environment setup._ | _Add the export command and artifact path._ | _Add the deterministic reference-data command and output directory._ | _List the files required by C++ tests._ |

`train_mnist.py` downloads the MNIST dataset into `python/mnist/data/` on its
first run.  Omit `--skip-onnx` when an optional ONNX export is also wanted.
After generating the MNIST files, build and run the C++ tests with
`./scripts/build.sh -r unittests` (or `-r tasks` for the task integration
tests).  Generated model artifacts and reference data are ignored by Git and
can be recreated with the commands above.

## Build and run

```bash
git submodule update --init
./scripts/build.sh
# Explicitly select unit tests:
./scripts/build.sh -r unittests
# Or run task integration tests:
./scripts/build.sh -r tasks
```

The script runs unit tests by default. Use `-r unittests` or `-r tasks` to
explicitly choose a test group. It configures and builds the selected test target,
then runs it directly. It exits with a non-zero status if any step fails.
