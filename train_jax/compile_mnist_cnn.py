"""python -m train_jax.compile_mnist_dnn"""

from absl import app
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

from . import compilation_utils

BATCH_SIZE = 32
IMAGE_SHAPE = (28, 28, 1)
CLASSES = 10


class CNN(nn.Module):

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

    x = nn.Dense(features=CLASSES)(x)
    x = nn.log_softmax(x)
    return x


def cross_entropy(variables, batch):
  inputs, labels = batch
  labels = jax.nn.one_hot(labels, 10)
  logits = CNN().apply(variables, inputs)
  return -jnp.mean(jnp.sum(logits * labels, axis=1))


def update(optimizer, batch):
  loss, gradient = jax.value_and_grad(cross_entropy)(optimizer.target, batch)
  optimizer = optimizer.apply_gradient(gradient)
  return optimizer, loss


def main(argv):
  images, labels = compilation_utils.get_random_data(BATCH_SIZE, IMAGE_SHAPE,
                                                     CLASSES)
  module = CNN()
  variables = module.init(jax.random.PRNGKey(0), images)

  compilation_utils.compile_apply(model_name="mnist_cnn",
                                  model_variables=variables,
                                  apply=module.apply,
                                  images=images)

  compilation_utils.compile_update(model_name="mnist_cnn",
                                   model_variables=variables,
                                   update=update,
                                   images=images,
                                   labels=labels)


if __name__ == "__main__":
  app.run(main)
