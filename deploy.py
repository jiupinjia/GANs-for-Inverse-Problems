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
import cv2
import initializer as init

# All parameters used in this file
Params = init.TrainingParamInitialization()



if os.path.exists(Params.result_dir) is False:
    os.mkdir(Params.result_dir)

if Params.task_name in ['unmixing_mnist_mnist', 'denoising', 'captcha']:
    n_channels = 1
else:
    n_channels = 3

input_shape = [None, Params.IMG_SIZE, Params.IMG_SIZE, n_channels]
Y = tf.placeholder(tf.float32, shape=input_shape)
Z = model.generator(Y, is_training=False)

# initialize the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# load the detection model
utils.load_historical_model(sess, Params.checkpoint_dir)


if not os.path.exists(Params.result_dir):
    os.makedirs(Params.result_dir)

m_test_samples = 1000



if Params.task_name is 'unmixing_mnist_cifar':

    data_cifar = utils.load_and_resize_cifar10_data(is_training=False)
    data_mnist = utils.load_and_resize_mnist_data(is_training=False)

    for i in range(m_test_samples):

        img_mnist = data_mnist[i,:,:,:]
        img_cifar = data_cifar[i,:,:,:]

        img_mixed = img_mnist + img_cifar

        img_mixed_ = np.expand_dims(img_mixed, axis=0)

        z_output = sess.run(Z, feed_dict={Y: img_mixed_})
        x_output = img_mixed_ - z_output
        x_output = x_output[0,:,:,:]
        z_output = z_output[0,:,:,:]

        save_path = os.path.join(
            Params.result_dir, '{}_img_mixed.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_mixed/img_mixed.max())

        save_path = os.path.join(
            Params.result_dir, '{}_mnist_output.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, z_output)

        save_path = os.path.join(
            Params.result_dir, '{}_mnist_true.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_mnist)

        save_path = os.path.join(
            Params.result_dir, '{}_cifar_output.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, x_output)

        save_path = os.path.join(
            Params.result_dir, '{}_cifar_true.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_cifar)

        print('processing %d-th image...' % i)



if Params.task_name is 'unmixing_mnist_mnist':

    data_mnist = utils.load_and_resize_mnist_data(is_training=False)

    for i in range(m_test_samples):

        img_mnist_1 = data_mnist[i,:,:,:]
        img_mnist_2 = data_mnist[-i,:,:,:]

        img_mixed = img_mnist_1 + img_mnist_2

        img_mixed_ = np.expand_dims(img_mixed, axis=0)

        z_output = sess.run(Z, feed_dict={Y: img_mixed_})
        x_output = img_mixed_ - z_output

        x_output[x_output<0] = 0

        x_output = x_output[0,:,:,0]
        z_output = z_output[0,:,:,0]
        img_mixed = img_mixed_[0,:,:,0]
        img_mnist_1 = img_mnist_1[:,:,0]
        img_mnist_2 = img_mnist_2[:,:,0]

        save_path = os.path.join(
            Params.result_dir, '{}_img_mixed.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_mixed/img_mixed.max(), cmap='gray')

        save_path = os.path.join(
            Params.result_dir, '{}_mnist_output_1.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, z_output, cmap='gray')

        save_path = os.path.join(
            Params.result_dir, '{}_mnist_output_2.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, x_output, cmap='gray')

        save_path = os.path.join(
            Params.result_dir, '{}_mnist_true_1.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_mnist_1, cmap='gray')

        save_path = os.path.join(
            Params.result_dir, '{}_mnist_true_2.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_mnist_2, cmap='gray')

        print('processing %d-th image...' % i)



if Params.task_name is 'denoising':

    data_mnist = utils.load_and_resize_mnist_data(is_training=False)

    for i in range(m_test_samples):

        img_mnist = data_mnist[i, :, :, :]
        img_mnist_ = np.expand_dims(img_mnist, axis=0)

        z_ = 1.0 * np.random.randn(1, Params.IMG_SIZE, Params.IMG_SIZE, 1)
        img_ns_ = img_mnist_ + z_

        z_output = sess.run(Z, feed_dict={Y: img_ns_})
        x_output = img_ns_ - z_output

        save_path = os.path.join(
            Params.result_dir, '{}_img_ns.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_ns_[0,:,:,0])

        save_path = os.path.join(
            Params.result_dir, '{}_img_gt.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, img_mnist_[0,:,:,0])

        save_path = os.path.join(
            Params.result_dir, '{}_img_output.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, x_output[0,:,:,0])

        save_path = os.path.join(
            Params.result_dir, '{}_z_output.png'.format(str(i).zfill(5)))
        plt.imsave(save_path, z_output[0,:,:,0])

        print('processing %d-th image...' % i)



if Params.task_name is 'captcha':

    data_x, data_y, labels = utils.load_captcha(
        m_samples=m_test_samples, color=True)

    for i in range(m_test_samples):

        img_captcha = data_y[i, :, :, :]
        x_output = np.zeros_like(img_captcha)
        for jj in range(3):
            single_band = img_captcha[:, :, jj]
            single_band = np.expand_dims(single_band, axis=-1)
            single_band = np.expand_dims(single_band, axis=0)
            z_output = sess.run(Z, feed_dict={Y: single_band})
            z_output = z_output[0, :, :, 0]
            x_output[:, :, jj] = img_captcha[:, :, jj] - z_output

        save_path = os.path.join(
            Params.result_dir, '{}_img_ns.png'.format(str(i).zfill(5)))
        f = 1 - cv2.resize(img_captcha, (160, 60))
        plt.imsave(save_path, f)

        save_path = os.path.join(
            Params.result_dir, '{}_img_output.png'.format(str(i).zfill(5)))
        f = 1 - cv2.resize(x_output, (160, 60))
        plt.imsave(save_path, f)

        save_path = os.path.join(
            Params.result_dir, '{}_z_output.png'.format(str(i).zfill(5)))
        f = 1 - cv2.resize((img_captcha - x_output), (160, 60))
        plt.imsave(save_path, f)

        print('processing %d-th image...' % i)

print('done.')


