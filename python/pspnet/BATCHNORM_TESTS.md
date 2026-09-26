# BatchNorm2D inference references

Generate separate float32 references using the cached pretrained PSPNet model:

```bash
python/vkai_py/bin/python python/pspnet/export_batchnorm_reference.py --image /path/to/scene.jpg
./scripts/build.sh -r unittests
./scripts/build.sh -r tasks
```

The exporter writes only `test_data/batchnorm/` (already ignored by Git). It does
not regenerate the existing Add or Conv2D fixtures. The tests read the exported
NCHW shape and epsilon rather than assuming a particular scene aspect ratio.

Two PSPNet BN layers are captured in eval mode before downstream in-place ReLU
and residual Add operations. Each fixture contains input, output, running mean,
running variance, gamma, beta, shape, and epsilon. Additional deterministic
PyTorch fixtures cover batch size 2, a dispatch tail, non-default epsilon, zero
variance, negative/zero gamma, and non-affine normalization.

`vkai::BatchNorm2D` supports contiguous NCHW float32 inference with fixed running
statistics. Empty gamma or beta means one or zero respectively. Training and
batch-statistics normalization (`track_running_stats=False`) are not supported.
Parameters are copied and converted to per-channel scale/offset at construction.
Comparison uses `1e-5 + 1e-5 * abs(reference)` to allow floating-point reassociation.
