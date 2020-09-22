"""
The test code for:
"Adversarial Training for Solving Inverse Problems in Image Processing"
using Tensorflow.

With this project, you can train a model to solve the following
inverse problems:
- on MNIST and CIFAR-10 datasets for separating superimposed images.
- image denoising on MNIST
- remove speckle and streak noise in CAPTCHAs
All the above tasks are trained w/ or w/o the help of pair-wise supervision.

"""

import tensorflow as tf
import utils
import model
import initializer as init

# All parameters used in this script
Params = init.TrainingParamInitialization()

if Params.task_name in ['unmixing_mnist_mnist', 'denoising', 'captcha']:
    n_channels = 1
else:
    n_channels = 3

# load mnist and cifar images
data = utils.load_data(is_training=True)

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1000000000.0, params.total_parameters))

Y_mb = utils.get_batch(data, Params.batch_size, 'Y_domain')
with tf.Graph().as_default() as graph:
    input_shape = [1, Params.IMG_SIZE, Params.IMG_SIZE, n_channels]
    Y = tf.placeholder(tf.float32, shape=input_shape)

    # generator statistics
    Z = model.generator(Y, is_training=True)

    # discriminator statistics
    # D2_logits, D2_prob = model.discriminator2(Y, is_training=True)

    stats_graph(graph)



