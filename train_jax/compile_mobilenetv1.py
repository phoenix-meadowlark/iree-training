"""python -m train_jax.compile_mnist_dnn"""

import pyiree as iree
import pyiree.jax

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np

import train_jax
from train_jax import mobilenetv1

BATCH_SIZE = 1
IMAGE_SHAPE = (224, 224, 3)
CLASSES = 1000


def cross_entropy(variables, batch):
  inputs, labels = batch
  labels = jax.nn.one_hot(labels, CLASSES)
  logits = mobilenetv1.MobileNetV1().apply(variables, inputs)
  return -jnp.mean(jnp.sum(logits * labels, axis=1))


def update(optimizer, batch):
  loss, gradient = jax.value_and_grad(cross_entropy)(optimizer.target, batch)
  optimizer = optimizer.apply_gradient(gradient)
  return optimizer, loss


def main():
  images, labels = train_jax.get_random_data(BATCH_SIZE, IMAGE_SHAPE, CLASSES)
  module = mobilenetv1.MobileNetV1(classes=CLASSES)
  variables = module.init(jax.random.PRNGKey(0), images)

  train_jax.compile_apply(model_name="mobilenetv1",
                          model_variables=variables,
                          apply=module.apply,
                          images=images)

  train_jax.compile_update(model_name="mobilenetv1",
                           model_variables=variables,
                           update=update,
                           images=images,
                           labels=labels)


if __name__ == "__main__":
  main()
