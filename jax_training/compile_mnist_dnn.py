"""python -m jax_training.compile_mnist_dnn"""

import pyiree as iree
import pyiree.jax

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

import jax_training


class Network(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=128)(x)
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
  images, labels = jax_training.get_random_data((32, 28, 28, 1))
  jax_training.compile_model(model_class=Network,
                             model_name="mnist_dnn",
                             update=update,
                             images=images,
                             labels=labels)


if __name__ == "__main__":
  main()
