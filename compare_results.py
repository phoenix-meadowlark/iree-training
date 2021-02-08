import numpy as np

from absl import flags
from absl import app

flags.DEFINE_string("path", None, "Path to the .expected and .result files.")
FLAGS = flags.FLAGS


def mlir_type_to_shape(mlir_type):
  shape = mlir_type.split("x")[:-1]
  if shape == [""]:
    return (1,)
  else:
    return tuple(map(int, shape))


def parse_arrays(path):
  with open(path, "r") as f:
    lines = f.readlines()

  # Validate and clean the output
  if "EXEC @main" in lines[0]:
    lines = lines[1:]
  for i, line in enumerate(lines):
    if "[...]" in line:
      raise ValueError(
          f"iree-run-module truncated output for shape {line.split('=')[0]}")
    line = line.replace("\n", "")
    line = line.replace("[", " ")
    line = line.replace("]", " ")
    lines[i] = line

  mlir_types = [line.split("=")[0] for line in lines]
  element_counts = [
      np.prod(mlir_type_to_shape(mlir_type)) for mlir_type in mlir_types
  ]
  values = [line.split("=")[1] for line in lines]

  arrays = []
  for value, count in zip(values, element_counts):
    arrays.append(np.fromstring(value, sep=" ", count=int(count)))
  return mlir_types, arrays


def main(argv):
  del argv  # Unused.

  _, compiled_results = parse_arrays(f"{FLAGS.path}.result")
  mlir_types, expected_results = parse_arrays(f"{FLAGS.path}.expected")

  for mlir_type, compiled, expected in zip(mlir_types, compiled_results,
                                           expected_results):
    abs_diff = np.max(np.abs(compiled - expected))
    warning = "****" if abs_diff > 1e-4 else ""
    print(f"{mlir_type:14s} {abs_diff:.3e} {warning}")
    # np.testing.assert_allclose(compiled, expected, 1e-6, 1e-6)


if __name__ == "__main__":
  app.run(main)
