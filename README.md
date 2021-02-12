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

### Compile Training and Inference

```shell
python -m train_jax.compile_mnist_dnn
adb push /tmp/iree-training/ "${atmp?}"
```

### Numerically verify

```shell
# apply
model="mnist_dnn"
adb shell "${atmp?}/iree/tools/iree-run-module" \
  -module_file="${atmp?}/iree-training/apply/${model?}.vmfb" \
  -function_inputs_file="${atmp?}/iree-training/apply/${model?}.data" \
  -driver=dylib \
  -entry_function=main > "/tmp/iree-training/apply/${model?}.result"
python ../iree-training/compare_results.py --path="/tmp/iree-training/apply/${model?}"
```

Optimizer can be any of `Adam`, `Adagrad`, `GradientDescent`, `LAMB`, `LARS`,
`Momentum`, or `RMSProp`.

```shell
# update
model="mnist_dnn"
optimizer="Adam"
adb shell "${atmp?}/iree/tools/iree-run-module" \
  -module_file="${atmp?}/iree-training/update/${model?}/${optimizer?}.vmfb" \
  -function_inputs_file="${atmp?}/iree-training/update/${model?}/${optimizer?}.data" \
  -driver=dylib \
  -entry_function=main > "/tmp/iree-training/update/${model?}/${optimizer?}.result"
python ../iree-training/compare_results.py --path="/tmp/iree-training/update/${model?}/${optimizer?}"
```

### Benchmark

```shell
# apply
model="mnist_dnn"
adb shell "${atmp?}/iree/tools/iree-benchmark-module" \
  -module_file="${atmp?}/iree-training/apply/${model?}.vmfb" \
  -function_inputs_file="${atmp?}/iree-training/apply/${model?}.mlir_types" \
  -driver=dylib \
  -entry_function=main
```

Optimizer can be any of `Adam`, `Adagrad`, `GradientDescent`, `LAMB`, `LARS`,
`Momentum`, or `RMSProp`.

```shell
# update
model="mnist_dnn"
optimizer="Adam"
adb shell "${atmp?}/iree/tools/iree-benchmark-module" \
  -module_file="${atmp?}/iree-training/update/${model?}/${optimizer?}.vmfb" \
  -function_inputs_file="${atmp?}/iree-training/update/${model?}/${optimizer?}.mlir_types" \
  -driver=dylib \
  -entry_function=main
```
