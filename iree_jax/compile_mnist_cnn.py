"""python -m iree_jax.compile_mnist_dnn"""

import pyiree as iree
import pyiree.jax

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

import iree_jax


class Network(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)

    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


def cross_entropy(variables, batch):
  inputs, labels = batch
  labels = jax.nn.one_hot(labels, 10)
  logits = Network().apply(variables, inputs)
  return -jnp.mean(jnp.sum(logits * labels, axis=1))


def update(optimizer, batch):
  loss, gradient = jax.value_and_grad(cross_entropy)(optimizer.target, batch)
  optimizer = optimizer.apply_gradient(gradient)
  return optimizer, loss


def main():
  images, labels = iree_jax.get_random_data((32, 28, 28, 1))
  iree_jax.compile_model(model_class=Network,
                             model_name="mnist_cnn",
                             update=update,
                             images=images,
                             labels=labels)


if __name__ == "__main__":
  main()