"""Things that should probably be in flax.linen in some form."""
from typing import *

import jax
import jax.numpy as jnp
import flax.linen as nn


def global_average_pooling(x):
  return jnp.mean(x, axis=(1, 2))


def flatten(x):
  return x.reshape((x.shape[0], -1))


class ZeroPad2D(nn.Module):
  padding: Union[int, Sequence[int], Sequence[Sequence[int]]]

  @nn.compact
  def __call__(self, inputs):
    if isinstance(self.padding, int):
      padding = [self.padding, self.padding]
    else:
      padding = list(self.padding)  # Convert to list for mutation.

    # If an int is given for a dimension's padding, apply it symmetrically.
    for dim, pad in enumerate(padding):
      if isinstance(pad, int):
        padding[dim] = [pad, pad]

    # Don't pad the batch dim or features dim.
    padding = [[0, 0]] + padding + [[0, 0]]
    return jnp.pad(inputs, padding)


class DepthwiseConv2D(nn.Module):
  """
  Based on haiku.DepthwiseConv2D:
  https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/depthwise_conv.py
  and flax.linen.Conv:
  https://github.com/google/flax/blob/master/flax/linen/linear.py#L194-L287
  """
  kernel_size: Sequence[int]
  strides: Sequence[int] = (1, 1)
  depth_multiplier: int = 1
  padding: Union[str, Sequence[Tuple[int, int]]] = "SAME"
  use_bias: bool = True
  dtype: jax.numpy.dtype = jnp.float32
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    features = int(self.depth_multiplier * inputs.shape[-1])
    kernel = self.param("kernel", self.kernel_init,
                        self.kernel_size + (1, features))
    kernel = jnp.asarray(kernel, self.dtype)

    y = jax.lax.conv_general_dilated(inputs,
                                     kernel,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     lhs_dilation=(1,) * len(self.kernel_size),
                                     rhs_dilation=(1,) * len(self.kernel_size),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                     feature_group_count=inputs.shape[-1])

    if self.use_bias:
      bias = self.param("bias", self.bias_init, (features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias

    return y
