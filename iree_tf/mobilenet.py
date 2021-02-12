import pyiree as iree
import pyiree.rt
import pyiree.compiler2.tf as tfc
from pyiree.tf.support import module_utils

import tensorflow as tf

class TrainableMobileNet(tf.Module):

  def __init__(self):
    self.mobilenet = tf.keras.applications.MobileNet(alpha=0.25, dropout=0)
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.SGD()

  @tf.function(input_signature=[
      tf.TensorSpec([1, 224, 224, 3], tf.float32),
      tf.TensorSpec([1], tf.float32)
  ])
  def train_on_batch(self, inputs, labels):
    with tf.GradientTape() as tape:
      probs = self.mobilenet(inputs, training=True)
      loss = self.loss(labels, probs)
    variables = self.mobilenet.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return loss

module = module_utils.IreeCompiledModule.create_from_class(
  TrainableMobileNet,
  module_utils.BackendInfo("iree_vmla"),
  exported_names=["train_on_batch"],
)
