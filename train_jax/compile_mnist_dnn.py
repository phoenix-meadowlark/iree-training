"""python -m train_jax.compile_mnist_dnn"""

import pyiree as iree
import pyiree.jax

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

import train_jax

BATCH_SIZE = 32
IMAGE_SHAPE = (28, 28, 1)
CLASSES = 10


class DNN(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)
    x = nn.Dense(features=CLASSES)(x)
    x = nn.log_softmax(x)
    return x


def cross_entropy(variables, batch):
  inputs, labels = batch
  labels = jax.nn.one_hot(labels, CLASSES)
  logits = DNN().apply(variables, inputs)
  return -jnp.mean(jnp.sum(logits * labels, axis=1))


def update(optimizer, batch):
  loss, gradient = jax.value_and_grad(cross_entropy)(optimizer.target, batch)
  optimizer = optimizer.apply_gradient(gradient)
  return optimizer, loss


def main():
  images, labels = train_jax.get_random_data(BATCH_SIZE, IMAGE_SHAPE, CLASSES)
  module = DNN()
  variables = module.init(jax.random.PRNGKey(0), images)

  train_jax.compile_apply(model_name="mnist_dnn",
                          model_variables=variables,
                          apply=module.apply,
                          images=images)

  train_jax.compile_update(model_name="mnist_dnn",
                           model_variables=variables,
                           update=update,
                           images=images,
                           labels=labels)


if __name__ == "__main__":
  main()
