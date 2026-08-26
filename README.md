# VKAI

A minimal C++ inference demo using FlatBuffers. The build converts fake model
weights from `assets/weights.json` into `build/weights.bin`; the executable maps
that file with `mmap`, accesses its float32 tensors in place, and runs a small
linear layer. The Protobuf dependency and schema remain in the repository but
are not currently built or used.

The MNIST training, generated artifacts, and C++ reference-data workflow live
in [`python/mnist`](python/mnist/README.md).

## Build and run

```bash
git submodule update --init
./build.sh
```

The script configures and builds the project, then runs the FlatBuffers `main`
executable directly. It exits with a non-zero status if any step fails.
