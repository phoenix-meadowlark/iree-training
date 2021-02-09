### Build IREE to target Android

`clean_build_android.sh` assumes that IREE is cloned below this repo's parent
directory.

```shell
./clean_build_android.sh
```

Copy over the tools:

```shell
export atmp="/data/local/tmp"
adb shell mkdir "${atmp?}/iree/"
adb push ../iree-build-android/iree/tools/ "${atmp?}/iree/"
```

### Compile Training

```shell
python -m iree_jax.compile_mnist_dnn
adb push /tmp/iree/training "${atmp?}/iree/"
```

### Numerically verify

Optimizer can be any of `Adam`, `Adagrad`, `GradientDescent`, `LAMB`, `LARS`,
`Momentum`, or `RMSProp`.

```shell
model="mnist_cnn"
optimizer="Adam"
adb shell "${atmp?}/iree/tools/iree-run-module" \
  -module_file="${atmp?}/iree/training/${model?}/${optimizer?}.vmfb" \
  -function_inputs_file="${atmp?}/iree/training/${model?}/${optimizer?}.data" \
  -driver=dylib \
  -entry_function=main > "/tmp/iree/training/${model?}/${optimizer?}.result"
python ../training/compare_results.py --path="/tmp/iree/training/${model?}/${optimizer?}"
```

### Benchmark

```shell
model="mnist_dnn"
optimizer="Adam"
adb shell "${atmp?}/iree/tools/iree-benchmark-module" \
  -module_file="${atmp?}/iree/training/${model?}/${optimizer?}.vmfb" \
  -function_inputs_file="${atmp?}/iree/training/${model?}/${optimizer?}.mlir_types" \
  -driver=dylib \
  -entry_function=main
```
