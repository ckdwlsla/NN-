import tensorflow as tf
import matplotlib.pyplot as plt


# Hyperparameter
initializer = tf.truncated_normal_initializer(stddev=0.02)
learning rate = 0.0002
batch_size = 256
epoch = 100000
lambda = 10

# Read image files
