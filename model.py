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
import tensorflow.contrib.slim as slim
import initializer as init

# All parameters used in this file
Params = init.TrainingParamInitialization()


def generator(Y, is_training):

    # define the number of filters in each conv layer
    mm = 64

    if Params.task_name in ['unmixing_mnist_mnist',
                            'denoising',
                            'captcha']:
        n_channels = 1
    else:
        n_channels = 3

    G_input_size = Params.G_input_size

    with tf.variable_scope('G_scope'):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer = tf.contrib.layers.xavier_initializer(),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}
                            ):

            Y_ = tf.image.resize_images(
                images=Y, size=[G_input_size, G_input_size])

            # encoder
            net_e_1 = slim.repeat(
                Y_, 2, slim.conv2d, mm, [4, 4], scope='e_conv1')
            net_e_1 = slim.conv2d(net_e_1, mm, [4,4], stride=2)

            net_e_2 = slim.repeat(
                net_e_1, 2, slim.conv2d, mm, [4, 4], scope='e_conv2')
            net_e_2 = slim.conv2d(net_e_2, mm, [4, 4], stride=2)

            net_e_3 = slim.repeat(
                net_e_2, 2, slim.conv2d, mm, [4, 4], scope='e_conv3')
            net_e_3 = slim.conv2d(net_e_3, mm, [4, 4], stride=2)

            net_e_4 = slim.repeat(
                net_e_3, 2, slim.conv2d, mm, [4, 4], scope='e_conv4')
            net4 = slim.conv2d(net_e_4, mm, [4, 4], stride=2)


            # decoder
            net_d_1 = slim.repeat(
                net4, 2, slim.conv2d, mm, [4, 4], scope='d_conv1')
            net_d_1 = tf.image.resize_images(
                images=net_d_1, size=[int(G_input_size / 8), int(G_input_size / 8)])
            net_d_1 = net_d_1 + net_e_3

            net_d_2 = slim.repeat(
                net_d_1, 2, slim.conv2d, mm, [4, 4], scope='d_conv2')
            net_d_2 = tf.image.resize_images(
                images=net_d_2, size=[int(G_input_size / 4), int(G_input_size / 4)])
            net_d_2 = net_d_2 + net_e_2

            net_d_3 = slim.repeat(
                net_d_2, 2, slim.conv2d, mm, [4, 4], scope='d_conv3')
            net_d_3 = tf.image.resize_images(
                images=net_d_3, size=[int(G_input_size / 2), int(G_input_size / 2)])
            net_d_3 = net_d_3 + net_e_1

            net_d_4 = slim.repeat(
                net_d_3, 2, slim.conv2d, mm, [4, 4], scope='d_conv4')
            net_d_4 = tf.image.resize_images(
                images=net_d_4, size=[int(G_input_size), int(G_input_size)])

            if Params.task_name in ['unmixing_mnist_cifar',
                                    'unmixing_mnist_mnist',
                                    'denoising']:
                net_d_4 = net_d_4 + tf.reduce_mean(Y_, axis=-1, keepdims=True)

            if Params.task_name in ['unmixing_mnist_cifar',
                                    'unmixing_mnist_mnist',
                                    'captcha']:
                net_d_4 = slim.repeat(
                    net_d_4, 2, slim.conv2d, mm, [4, 4], scope='d_conv5')

            F_ = slim.conv2d(net_d_4, n_channels, [4, 4], activation_fn=tf.nn.relu)
            F_ = tf.image.resize_images(
                images=F_, size=[Params.IMG_SIZE, Params.IMG_SIZE],
                method=tf.image.ResizeMethod.BICUBIC)

            if Params.task_name is 'denoising':
                Z = Y_ - F_
            else:
                Z = F_

    return Z




def discriminator1(x, is_training, reuse=False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope('D1_scope'):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            activation_fn=tf.nn.leaky_relu,
                            ):

            x = tf.image.resize_images(
                images=x, size=[Params.D_input_size, Params.D_input_size])

            net = slim.conv2d(x, 32, [4, 4], stride=2)

            net = slim.repeat(
                net, 2, slim.conv2d, 64, [4, 4], scope='conv2')
            net = slim.conv2d(net, 64, [4, 4], stride=2)

            net = slim.repeat(
                net, 2, slim.conv2d, 128, [4, 4], scope='conv3')
            net = slim.conv2d(net, 128, [4, 4], stride=2)

            net = slim.repeat(
                net, 2, slim.conv2d, 256, [4, 4], scope='conv4')
            net = slim.conv2d(net, 256, [4, 4], stride=2)

            net = slim.flatten(net)
            D_logit = slim.fully_connected(net, 1, activation_fn=None)
            D_prob = tf.nn.sigmoid(D_logit)

    return D_logit, D_prob




def discriminator2(x, is_training, reuse=False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope('D2_scope'):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            activation_fn=tf.nn.leaky_relu,
                            ):
            x = tf.image.resize_images(
                images=x, size=[Params.D_input_size, Params.D_input_size])

            net = slim.conv2d(x, 32, [4, 4], stride=2)

            net = slim.repeat(
                net, 2, slim.conv2d, 64, [4, 4], scope='conv2')
            net = slim.conv2d(net, 64, [4, 4], stride=2)

            net = slim.repeat(
                net, 2, slim.conv2d, 128, [4, 4], scope='conv3')
            net = slim.conv2d(net, 128, [4, 4], stride=2)

            net = slim.repeat(
                net, 2, slim.conv2d, 256, [4, 4], scope='conv4')
            net = slim.conv2d(net, 256, [4, 4], stride=2)

            net = slim.flatten(net)
            D_logit = slim.fully_connected(net, 1, activation_fn=None)
            D_prob = tf.nn.sigmoid(D_logit)

    return D_logit, D_prob



def compute_loss(D1_logits_real, D1_prob_real,
                 D2_logits_real, D2_prob_real,
                 D1_logits_fake, D1_prob_fake,
                 D2_logits_fake, D2_prob_fake):

    if Params.gan_model is 'W_GAN':
        D1_loss = - (tf.reduce_mean(D1_logits_real)
                     - tf.reduce_mean(D1_logits_fake))
        D2_loss = - (tf.reduce_mean(D2_logits_real)
                     - tf.reduce_mean(D2_logits_fake))
        if Params.with_random_dual_mapping:
            G_loss = - tf.reduce_mean(D1_logits_fake + D2_logits_fake)
        else:
            G_loss = - tf.reduce_mean(D1_logits_fake)
        tf.summary.scalar('D_loss', D1_loss + D2_loss)
        tf.summary.scalar('D1_loss', D1_loss)
        tf.summary.scalar('D2_loss', D2_loss)
        tf.summary.scalar('G_loss', G_loss)

    if Params.gan_model is 'LS_GAN':
        D1_loss = 0.5 * (tf.reduce_mean((D1_logits_real - 1) ** 2) +
                         tf.reduce_mean(D1_logits_fake ** 2))
        D2_loss = 0.5 * (tf.reduce_mean((D2_logits_real - 1) ** 2) +
                         tf.reduce_mean(D2_logits_fake ** 2))
        if Params.with_random_dual_mapping:
            G_loss = 0.5 * tf.reduce_mean((D1_logits_fake - 1) ** 2 +
                                          (D2_logits_fake - 1) ** 2)
        else:
            G_loss = 0.5 * tf.reduce_mean((D1_logits_fake - 1) ** 2)
        tf.summary.scalar('D_loss', D1_loss + D2_loss)
        tf.summary.scalar('D1_loss', D1_loss)
        tf.summary.scalar('D2_loss', D2_loss)
        tf.summary.scalar('G_loss', G_loss)


    if Params.gan_model is 'Vanilla_GAN':
        D1_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D1_logits_real, labels=tf.ones_like(D1_logits_real)))
        D1_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D1_logits_fake, labels=tf.zeros_like(D1_logits_fake)))
        D1_loss = D1_loss_real + D1_loss_fake

        D2_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D2_logits_real, labels=tf.ones_like(D2_logits_real)))
        D2_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D2_logits_fake, labels=tf.zeros_like(D2_logits_fake)))
        D2_loss = D2_loss_real + D2_loss_fake

        if Params.with_random_dual_mapping:
            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D1_logits_fake, labels=tf.ones_like(D1_logits_fake)) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D2_logits_fake, labels=tf.ones_like(D2_logits_fake))
            )
        else:
            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D1_logits_fake, labels=tf.ones_like(D1_logits_fake))
            )
        tf.summary.scalar('D_loss', D1_loss + D2_loss)
        tf.summary.scalar('D1_loss', D1_loss)
        tf.summary.scalar('D2_loss', D2_loss)
        tf.summary.scalar('G_loss', G_loss)

    return D1_loss, D2_loss, G_loss






def initilize_solvers(D1_loss, D2_loss, G_loss, D_steps, G_steps):

    if Params.gan_model is 'W_GAN':
        optimizer = 'RMSProp'
        d_clip = 0.01
    else: # Vanilla_GAN or LS_GAN
        optimizer = 'Adam'
        d_clip = 1e9

    tvars = tf.trainable_variables()
    theta_D1 = [var for var in tvars if 'D1_scope' in var.name]
    theta_D2 = [var for var in tvars if 'D2_scope' in var.name]
    theta_G = [var for var in tvars if 'G_scope' in var.name]

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

        if optimizer is 'RMSProp':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                D1_solver = (tf.train.RMSPropOptimizer(
                    learning_rate=Params.d_learning_rate).minimize(
                    D1_loss, var_list=theta_D1, global_step=D_steps))
                D2_solver = (tf.train.RMSPropOptimizer(
                    learning_rate=Params.d_learning_rate).minimize(
                    D2_loss, var_list=theta_D2, global_step=D_steps))
                G_solver = (tf.train.RMSPropOptimizer(
                    learning_rate=Params.g_learning_rate).minimize(
                    G_loss, var_list=theta_G, global_step=G_steps))

        if optimizer is 'Adam':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                D1_solver = (tf.train.AdamOptimizer(
                    learning_rate=Params.d_learning_rate).minimize(
                    D1_loss, var_list=theta_D1, global_step=D_steps))
                D2_solver = (tf.train.AdamOptimizer(
                    learning_rate=Params.d_learning_rate).minimize(
                    D2_loss, var_list=theta_D2, global_step=D_steps))
                G_solver = (tf.train.AdamOptimizer(
                    learning_rate=Params.g_learning_rate).minimize(
                    G_loss, var_list=theta_G, global_step=G_steps))

        clip_D1 = [p.assign(tf.clip_by_value(p, -d_clip, d_clip))
                   for p in theta_D1]
        clip_D2 = [p.assign(tf.clip_by_value(p, -d_clip, d_clip))
                   for p in theta_D2]

    return D1_solver, D2_solver, G_solver, clip_D1, clip_D2



def cpt_paired_loss(X_gen, X1, Y):

    if Params.task_name is 'unmixing_mnist_mnist':
        # To avoid ambiguity, when performing mnist-mnist
        # umixing, the residuals are computed individually
        # on two different components, and then the gradient
        # is computed based on their minimum value.
        loss_1 = 1000.*tf.reduce_mean(tf.abs(X_gen-X1))
        loss_2 = 1000.*tf.reduce_mean(tf.abs(X_gen-(Y-X1)))
        L2loss = tf.minimum(loss_1, loss_2)
    else:
        L2loss = 1000. * tf.reduce_mean(tf.abs(X_gen - X1))

    return L2loss