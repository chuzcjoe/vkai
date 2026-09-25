# VKAI

VKAI is a neural-network inference engine built on Vulkan. It aims to make
inference portable across platforms and GPU vendors by using Vulkan rather than
vendor-specific software ecosystems such as CUDA or HIP.

Its main building blocks are:

- [CORE](external/CORE), which provides the abstraction over the Vulkan API.
- [FlatBuffers](external/flatbuffers), which stores model weights in a compact,
  directly readable binary format.

The project loads FlatBuffer weight artifacts and executes neural-network
operators through Vulkan, without requiring a vendor-specific GPU runtime.

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
