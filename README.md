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

| Model | Generate inference artifact | Generate C++ test reference data | Generated files |
| --- | --- | --- | --- |
| MNIST (784 → 128 → 10) | `python train.py` | `python test.py --index 0` | `python/mnist/mnist_weights.bin` (FlatBuffer weights), `python/mnist/mnist_model.pth`, and `python/mnist/test_data/*.bin` plus `metadata.json` |

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
