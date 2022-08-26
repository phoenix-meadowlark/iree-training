"""python -m train_jax.compile_resnetv1"""

from absl import app
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

from . import compilation_utils
from . import resnetv1

BATCH_SIZE = 1
IMAGE_SHAPE = (224, 224, 3)
CLASSES = 1000


def cross_entropy(variables, batch):
  inputs, labels = batch
  labels = jax.nn.one_hot(labels, CLASSES)
  logits = resnetv1.ResNet18(num_classes=CLASSES).apply(variables, inputs)
  return -jnp.mean(jnp.sum(logits * labels, axis=1))


def update(optimizer, batch):
  loss, gradient = jax.value_and_grad(cross_entropy)(optimizer.target, batch)
  optimizer = optimizer.apply_gradient(gradient)
  return optimizer, loss


def main(argv):
  images, labels = compilation_utils.get_random_data(BATCH_SIZE, IMAGE_SHAPE,
                                                     CLASSES)
  module = resnetv1.ResNet18(num_classes=CLASSES)
  variables = module.init(jax.random.PRNGKey(0), images)

  compilation_utils.compile_apply(model_name="resnetv1_18",
                                  model_variables=variables,
                                  apply=module.apply,
                                  images=images)

  # compilation_utils.compile_update(model_name="resnetv1_18",
  #                                  model_variables=variables,
  #                                  update=update,
  #                                  images=images,
  #                                  labels=labels)


if __name__ == "__main__":
  app.run(main)
