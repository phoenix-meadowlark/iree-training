import itertools
import os
from typing import *

from absl import flags
# import iree as iree
# import iree.jax
# import iree.runtime

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

flags.DEFINE_string("base_dir", "/tmp/", "The base directory to store compiled "
                    "artifacts under.")
FLAGS = flags.FLAGS

__all__ = [
    "ANDROID_OPTIONS",
    "OPTIMIZERS_TO_HPARAMS",
    "compile_apply",
    "compile_update",
    "get_jax_mlir_types",
    "get_jax_serialized_data",
    "get_random_data",
]

TOLERANCES = dict(atol=1e-4, rtol=1e-4)

OPTIMIZERS_TO_HPARAMS = {
    "GradientDescent": dict(learning_rate=1e-2),
    "Adam": dict(learning_rate=1e-3, weight_decay=1e-1),
    "Adagrad": dict(learning_rate=1e-3),
    "LAMB": dict(learning_rate=1e-3, weight_decay=1e-1),
    "LARS": dict(learning_rate=1e-3, weight_decay=1e-1),
    "Momentum": dict(learning_rate=1e-3, weight_decay=1e-1),
    "RMSProp": dict(learning_rate=1e-3),
}

ANDROID_OPTIONS = {
    "target_backends": ["dylib-llvm-aot"],
    "extra_args": ["--iree-llvm-target-triple=aarch64-none-linux-android29"],
}

IREE_DEVICE = jax.devices("iree")[0]


def _numpy_dtype_to_mlir_element_type(dtype: np.dtype) -> str:
  """Returns a string that denotes the type 'dtype' in MLIR style."""
  if not isinstance(dtype, np.dtype):
    dtype = np.dtype(dtype)  # Handle np.int8 _not_ being a dtype.

  bits = dtype.itemsize * 8
  if np.issubdtype(dtype, np.integer):
    return f"i{bits}"
  elif np.issubdtype(dtype, np.floating):
    return f"f{bits}"
  else:
    raise TypeError(f"Expected integer or floating type, but got {dtype}")


def get_mlir_type(array: Any, allow_non_mlir_dtype: bool = True) -> str:
  array = iree.runtime.normalize_value(array)
  shape = "x".join([str(dim) for dim in array.shape])
  if np.issubdtype(array.dtype, np.number):
    element_type = _numpy_dtype_to_mlir_element_type(array.dtype)
  else:
    element_type = f"<dtype '{array.dtype}'>"
  return f"{shape}x{element_type}"


def get_jax_mlir_types(*args, **kwargs):
  args_flat, _ = jax.tree_flatten((args, kwargs))
  types = []
  for arg in args_flat:
    arg = iree.runtime.normalize_value(arg)
    types.append(get_mlir_type(arg))
  return types


def get_jax_serialized_data(*args, **kwargs):
  types = get_jax_mlir_types(*args, **kwargs)
  args_flat, _ = jax.tree_flatten((args, kwargs))
  data = []
  for arg, mlir_type in zip(args_flat, types):
    arg = np.ndarray.flatten(iree.runtime.normalize_value(arg))
    values = " ".join(str(v) for v in arg)
    data.append(f"{mlir_type}={values}")
  return data


def get_random_data(batch_size: int, image_shape: Tuple[int], classes: int):
  np.random.seed(0)
  batched_shape = (batch_size,) + image_shape
  images = np.random.uniform(0, 1, batched_shape).astype(np.float32)
  labels = np.random.choice(classes, batch_size).astype(np.int32)
  return images, labels


def compile_update(model_name, model_variables, update, images, labels):
  model_path = os.path.join(FLAGS.base_dir, "iree-training", model_name)
  os.makedirs(model_path, exist_ok=True)

  for opt_name, hparams in OPTIMIZERS_TO_HPARAMS.items():
    print(f"\nCompiling: {opt_name}")
    optimizer_def = getattr(flax.optim, opt_name)(**hparams)
    optimizer = optimizer_def.create(model_variables)
    args = [optimizer, [images, labels]]

    # # Save a OptName.mlir_types file that can be passed via `--function_inputs_file`
    # print("Getting mlir_types")
    # mlir_types = get_jax_mlir_types(*args)
    # with open(os.path.join(model_path, f"{opt_name}.mlir_types"), "w") as f:
    #   f.write("\n".join(mlir_types))

    # # Save a OptName.data file that can be passed via `--function_inputs_file`
    # # for numerical validation.
    # print("Getting serialized inputs")
    # data = get_jax_serialized_data(*args)
    # with open(os.path.join(model_path, f"{opt_name}.data"), "w") as f:
    #   f.write("\n".join(data))

    # Save a OptName.expected file that can be used to numerically verify IREE
    # running on Android.
    print("Getting expected results")
    expected_results = update(*args)
    # expected_data = get_jax_serialized_data(expected_results)
    # with open(os.path.join(model_path, f"{opt_name}.expected"), "w") as f:
    #   f.write("\n".join(expected_data))

    # Export the MLIR-HLO for debugging purposes.
    # print("Exporting MLIR-HLO")
    # iree.jax.aot(update,
    #              *args,
    #              import_only=True,
    #              output_file=os.path.join(model_path, f"{opt_name}.mlir"))

    # Validate the host execution correctness.
    print("Validating IREE host execution correctness")
    # iree_update = iree.jax.jit(update)
    host_results = jax.jit(update, device=IREE_DEVICE)(*args)
    host_values, _ = jax.tree_flatten(host_results)
    expected_values, _ = jax.tree_flatten(expected_results)
    for host_value, expected_value in zip(host_values, expected_values):
      print(np.max(np.abs(host_value - expected_value)))
      np.testing.assert_allclose(host_value, expected_value, **TOLERANCES)

    # Compile the model for Android and save the compiled flatbuffer.
    # print("Compiling for Android")
    # iree.jax.aot(update,
    #              *args,
    #              output_file=os.path.join(model_path, f"{opt_name}.vmfb"),
    #              **ANDROID_OPTIONS)


def compile_apply(model_name, model_variables, apply, images):
  print(f"\nCompiling {model_name}.apply")
  model_path = os.path.join(FLAGS.base_dir, "iree-training", model_name)
  os.makedirs(model_path, exist_ok=True)
  args = [model_variables, images]

  print("Getting expected results")
  expected_results = apply(*args)

  # Validate the host execution correctness.
  print("Validating IREE host execution correctness")
  # iree_apply = iree.jax.jit(apply)
  host_results = jax.jit(apply, device=IREE_DEVICE)(*args)
  host_values, _ = jax.tree_flatten(host_results)
  expected_values, _ = jax.tree_flatten(expected_results)
  for host_value, expected_value in zip(host_values, expected_values):
    print(np.max(np.abs(host_value - expected_value)))
    np.testing.assert_allclose(host_value, expected_value, **TOLERANCES)
