"""MobileNetV1 Implemented to Mirror the Keras Applications Implementation."""
# Based on:
#   https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/mobilenet.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import *
import functools

import jax
import flax.linen as nn

from train_jax import nn_util


class MobileNetV1Block(nn.Module):
  pointwise_conv_features: int
  alpha: float = 1.0
  depth_multiplier: int = 1
  strides: Tuple[int, int] = (1, 1)
  use_batch_norm: bool = False

  @nn.compact
  def __call__(self, x):
    if self.strides != (1, 1):
      x = nn_util.ZeroPad2D(((0, 1), (0, 1)))(x)

    x = nn_util.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=self.strides,
        depth_multiplier=self.depth_multiplier,
        padding="same" if self.strides == (1, 1) else "valid",
        use_bias=False)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(momentum=0.99, epsilon=1e-3)(x)
    x = jax.nn.relu6(x)

    x = nn.Conv(features=int(self.pointwise_conv_features * self.alpha),
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                use_bias=False)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(momentum=0.99, epsilon=1e-3)(x)
    return jax.nn.relu6(x)


class MobileNetV1(nn.Module):
  """MobileNetV1 Network.

  Forked from the tf.keras.applications.MobileNet implementation.

  Attributes:
    feature_multiplier:
      Controls the number of features used in nn.Conv layers. Referred to as the
      'width multiplier' in the paper, and 'alpha' in the Keras implementation.
    depth_multiplier:
      Controls the depth_multiplier for the nn.DepthwiseConv layers. Referred to
      as the 'resolution multiplier' in the paper.
    include_top:
      Whether or not to include the classification block after the main block.
    dropout:
      Controls the dropout in the classification block.
    classes:
      Controls the number of classes in the classification block.
    classifier_activation:
      The final activation to apply in the classification block.
    use_batch_norm:
      Whether or not to use nn.BatchNorm. Just here because I haven't spent the
      time to figure out how to set up the flax invocations.
  """
  alpha: float = 1.0
  depth_multiplier: float = 1.0
  include_top: bool = True
  dropout: float = 0
  classes: int = 1000
  classifier_activation: Callable = nn.log_softmax
  use_batch_norm: bool = False

  @nn.compact
  def __call__(self, x):
    # Input block.
    x = nn.Conv(features=int(32 * self.alpha),
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                use_bias=False)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(momentum=0.99, epsilon=1e-3)(x)
    x = jax.nn.relu6(x)

    # Main Network.
    mobilenetv1_block = functools.partial(
        MobileNetV1Block,
        alpha=self.alpha,
        depth_multiplier=self.depth_multiplier)

    x = mobilenetv1_block(pointwise_conv_features=64)(x)

    x = mobilenetv1_block(pointwise_conv_features=128, strides=(2, 2))(x)
    x = mobilenetv1_block(pointwise_conv_features=128)(x)

    x = mobilenetv1_block(pointwise_conv_features=256, strides=(2, 2))(x)
    x = mobilenetv1_block(pointwise_conv_features=256)(x)

    x = mobilenetv1_block(pointwise_conv_features=512, strides=(2, 2))(x)
    x = mobilenetv1_block(pointwise_conv_features=512)(x)
    x = mobilenetv1_block(pointwise_conv_features=512)(x)
    x = mobilenetv1_block(pointwise_conv_features=512)(x)
    x = mobilenetv1_block(pointwise_conv_features=512)(x)
    x = mobilenetv1_block(pointwise_conv_features=512)(x)

    x = mobilenetv1_block(pointwise_conv_features=1024, strides=(2, 2))(x)
    x = mobilenetv1_block(pointwise_conv_features=1024)(x)

    # Classification layers.
    if self.include_top:
      x = nn_util.global_average_pooling(x)
      x = x.reshape((-1, 1, 1, int(1024 * self.alpha)))
      x = nn.Dropout(self.dropout)(x)
      x = nn.Conv(features=self.classes, kernel_size=(1, 1), padding="same")(x)
      x = nn_util.flatten(x)
      x = self.classifier_activation(x)

    return x
