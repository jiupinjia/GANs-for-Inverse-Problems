"""
The test code for:
"Adversarial Training for Solving Inverse Problems"
using Tensorflow.

With this project, you can train a model to solve the following
inverse problems:
- on MNIST and CIFAR-10 datasets for separating superimposed images.
- image denoising on MNIST
- remove speckle and streak noise in CAPTCHAs
All the above tasks are trained w/ or w/o the help of pair-wise supervision.

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import model
import initializer as init

# All parameters used in this script
Params = init.TrainingParamInitialization()

if Params.task_name in ['unmixing_mnist_mnist', 'denoising', 'captcha']:
    n_channels = 1
else:
    n_channels = 3

if Params.gan_model is 'Vanilla_GAN':
    d_thresh = 0.5
elif Params.gan_model is 'LS_GAN':
    d_thresh = 0.1
elif Params.gan_model is 'W_GAN':
    d_thresh = -5
else:
    d_thresh = -1e9

input_shape = [None, Params.IMG_SIZE, Params.IMG_SIZE, n_channels]
Y = tf.placeholder(tf.float32, shape=input_shape)
X1 = tf.placeholder(tf.float32, shape=input_shape)
X2 = tf.placeholder(tf.float32, shape=input_shape)

# build the graph
Z = model.generator(Y, is_training=True)
X_gen = Y - Z
X_gen = tf.clip_by_value(X_gen, clip_value_min=0, clip_value_max=1e9)
Y_gen = X2 + Z
D1_logits_real, D1_prob_real = model.discriminator1(
    X1, is_training=True)
D2_logits_real, D2_prob_real = model.discriminator2(
    Y, is_training=True)
D1_logits_fake, D1_prob_fake = model.discriminator1(
    X_gen, is_training=True, reuse=True)
D2_logits_fake, D2_prob_fake = model.discriminator2(
    Y_gen, is_training=True, reuse=True)

# With pair-wise loss?
if Params.with_paired_loss:
    L2loss = model.cpt_paired_loss(X_gen, X1, Y)
else:
    L2loss = 0.

# compute the loss
D1_loss, D2_loss, G_loss = model.compute_loss(
    D1_logits_real, D1_prob_real, D2_logits_real, D2_prob_real,
    D1_logits_fake, D1_prob_fake, D2_logits_fake, D2_prob_fake)
G_loss = G_loss + L2loss

# to record the iteration number
G_steps = tf.Variable(
    0, trainable=False, name='G_steps')
D_steps = tf.Variable(
    0, trainable=False, name='D_steps')

# initialize the solvers
D1_solver, D2_solver, G_solver, clip_D1, clip_D2 = \
    model.initilize_solvers(
        D1_loss, D2_loss, G_loss, D_steps, G_steps)

# initialize the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# create a summary writer
merged_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('', sess.graph)

# load historical model and create a saver
utils.load_historical_model(sess, checkpoint_dir=Params.checkpoint_dir)
saver = tf.train.Saver()

if not os.path.exists(Params.sample_dir):
    os.makedirs(Params.sample_dir)

# load mnist and cifar images
data = utils.load_data(is_training=True)

g_iter = 0
d_iter = 0
D1_loss_val = 0
D2_loss_val = 0
mm = 1

while g_iter < Params.max_iters:

    if Params.with_paired_loss:
        X1_mb, Y_mb = utils.get_batch(data, Params.batch_size, 'XY_pair')
    else:
        X1_mb = utils.get_batch(data, Params.batch_size, 'X_domain')
        Y_mb = utils.get_batch(data, Params.batch_size, 'Y_domain')
    X2_mb = utils.get_batch(data, Params.batch_size, 'X_domain')

    _, D1_loss_val, _, summary = sess.run(
        [D1_solver, D1_loss, clip_D1, merged_op],
        feed_dict={X1: X1_mb, X2: X2_mb, Y: Y_mb})

    if Params.with_random_dual_mapping:
        _, D2_loss_val, _, summary = sess.run(
            [D2_solver, D2_loss, clip_D2, merged_op],
            feed_dict={X1: X1_mb, X2: X2_mb, Y: Y_mb})

    d_iter = sess.run(D_steps)
    # write states to summary
    summary_writer.add_summary(summary, g_iter)

    if D1_loss_val < d_thresh or D2_loss_val < d_thresh:
        mm = mm + 5
    else:
        mm = 1

    for _ in range(mm):

        if Params.with_paired_loss:
            X1_mb, Y_mb = utils.get_batch(data, Params.batch_size, 'XY_pair')
        else:
            X1_mb = utils.get_batch(data, Params.batch_size, 'X_domain')
            Y_mb = utils.get_batch(data, Params.batch_size, 'Y_domain')
        X2_mb = utils.get_batch(data, Params.batch_size, 'X_domain')

        _, G_loss_val, D1_loss_val, D2_loss_val = sess.run(
            [G_solver, G_loss, D1_loss, D2_loss],
                    feed_dict={X1: X1_mb, X2: X2_mb, Y: Y_mb})
        g_iter = sess.run(G_steps)

        # save generated samples
        if g_iter % 20 == 0:
            output_x, output_z = sess.run(
                [X_gen, Z], feed_dict={X1: X1_mb, X2: X2_mb, Y: Y_mb})

            save_path = os.path.join(
                Params.sample_dir, '{}_output_x.png'.format(str(g_iter).zfill(5)))
            plt.imsave(save_path, utils.plot2x2(output_x))

            save_path = os.path.join(
                Params.sample_dir, '{}_output_z.png'.format(str(g_iter).zfill(5)))
            plt.imsave(save_path, utils.plot2x2(output_z))

            save_path = os.path.join(
                Params.sample_dir, '{}_input_y.png'.format(str(g_iter).zfill(5)))
            if Params.task_name is 'unmixing':
                Y_mb = Y_mb/Y_mb.max()
            plt.imsave(save_path, utils.plot2x2(Y_mb))

        if g_iter % 20 == 0:
            print('D1_loss = %g, D2_loss = %g, G_loss = %g' %
                  (D1_loss_val, D2_loss_val, G_loss_val))
            print('g_iter = %d, d_iter = %d, n_g/d = %d' %
                  (g_iter, d_iter, mm))
            print()

        # save model every 500 g_iters
        if np.mod(g_iter, 500) == 1 and g_iter > 1:
            print('saving model to checkpoint ...')
            saver.save(sess, os.path.join(Params.checkpoint_dir, 'G_step'),
                       global_step=G_steps)

