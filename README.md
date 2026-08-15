# VKAI

A minimal C++ demo that builds FlatBuffers from a Git submodule, generates C++
code from a schema, and runs a serialization round trip. The Protobuf dependency
and schema remain in the repository but are not currently built or used.

## Build and run

```bash
git submodule update --init
./build.sh
```

The script configures and builds the project, then runs the FlatBuffers `main`
executable directly. It exits with a non-zero status if any step fails.
