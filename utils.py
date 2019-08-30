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

import numpy as np
import os
import tensorflow as tf
import cv2
import initializer as init
import string
characters = string.digits + string.ascii_uppercase
import random
from captcha.image import ImageCaptcha
Image = ImageCaptcha()

# All parameters used in this file
Params = init.TrainingParamInitialization()


def get_batch(DATA, batch_size, mode):

    if mode is 'X_domain':
        n, h, w, c = DATA['Source1'].shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        batch = DATA['Source1'][idx, :, :, :]

        return batch

    if mode is 'Y_domain':

        if Params.task_name in ['unmixing_mnist_cifar', 'unmixing_mnist_mnist']:

            n, h, w, c = DATA['Source1'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source1'][idx, :, :, :]

            # # image mixture
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            z = DATA['Source2'][idx, :, :, :]
            batch = batch + z

            return batch

        if Params.task_name is 'denoising':

            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source2'][idx, :, :, :]

            # # add noise
            z = 1.0 * np.random.randn(batch_size, h, w, c)
            batch = batch + z

            return batch

        if Params.task_name is 'captcha':

            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source2'][idx, :, :, :]

            return batch



    if mode is 'XY_pair':

        if Params.task_name in ['unmixing_mnist_cifar', 'unmixing_mnist_mnist']:
            n, h, w, c = DATA['Source1'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source1'][idx, :, :, :]

            # # image mixture
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            z = DATA['Source2'][idx, :, :, :]

            return batch, batch + z

        if Params.task_name is 'denoising':
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source2'][idx, :, :, :]

            # # add noise
            z = 1.0 * np.random.randn(batch_size, h, w, c)

            return batch, batch + z

        if Params.task_name is 'captcha':
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch_x = DATA['Source1'][idx, :, :, :]
            batch_y = DATA['Source2'][idx, :, :, :]

            return batch_x, batch_y





def plot2x2(samples):

    IMG_SIZE = samples.shape[1]

    if Params.task_name in ['unmixing_mnist_mnist', 'denoising']:
        n_channels = 1
    else:
        n_channels = 3

    img_grid = np.zeros((2 * IMG_SIZE, 2 * IMG_SIZE, n_channels))

    for i in range(4):
        py, px = IMG_SIZE * int(i / 2), IMG_SIZE * (i % 2)
        this_img = samples[i, :, :, :]
        img_grid[py:py + IMG_SIZE, px:px + IMG_SIZE, :] = this_img

    if n_channels == 1:
        img_grid = img_grid[:,:,0]

    if Params.task_name is 'captcha':
        img_grid = 1 - cv2.resize(img_grid, (320, 120))

    return img_grid




def load_historical_model(sess, checkpoint_dir='checkpoints'):

    # check and create model dir
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)

    if 'checkpoint' in os.listdir(checkpoint_dir):
        # training from the last checkpoint
        print('loading model from the last checkpoint ...')
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)
        print(latest_checkpoint)
        print('loading finished!')
    else:
        print('no historical model found, start training from scratch!')




def load_and_resize_mnist_data(is_training):

    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.mnist.load_data()
    if is_training is True:
        data = x_train/255.
    else:
        data = x_test/255.

    m = data.shape[0]
    data = np.reshape(data, [m, 28, 28])
    if Params.task_name in ['unmixing_mnist_mnist', 'denoising']:
        n_channels = 1
    else:
        n_channels = 3
    data_rs = np.zeros((m, Params.IMG_SIZE, Params.IMG_SIZE, n_channels))

    for i in range(m):
        img = data[i, :, :]
        img_rs = cv2.resize(img, dsize=(Params.IMG_SIZE, Params.IMG_SIZE))
        if Params.task_name in ['unmixing_mnist_mnist', 'denoising']:
            img_rs = np.expand_dims(img_rs, axis=-1)
        else:
            img_rs = np.stack([img_rs, img_rs, img_rs], axis=-1)
        data_rs[i, :, :, :] = img_rs

    return data_rs




def load_and_resize_cifar10_data(is_training):

    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.cifar10.load_data()

    if is_training is True:
        data = x_train/255.
    else:
        data = x_test/255.

    m = data.shape[0]
    n_channels = 3
    data_rs = np.zeros((m, Params.IMG_SIZE, Params.IMG_SIZE, n_channels))

    for i in range(m):
        img = data[i, :, :, :]
        img_rs = cv2.resize(img, dsize=(Params.IMG_SIZE, Params.IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        data_rs[i, :, :, :] = img_rs

    return data_rs




def load_captcha(m_samples=10000, color=False):

    if color is True:
        data_x = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 3))
        data_y = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 3))
    else:
        data_x = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 1))
        data_y = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 1))

    labels = []

    for i in range(m_samples):

        random_str = ''.join([random.choice(characters) for j in range(4)])
        band_id = np.random.randint(3)
        img_clean, img_ns = Image.generate_image_pair(random_str)

        img_clean = np.array(img_clean, dtype=np.uint8)
        img_clean = 1 - img_clean / 255.
        img_clean = cv2.resize(
            img_clean, (Params.IMG_SIZE, Params.IMG_SIZE))
        if color is not True:
            img_clean = np.expand_dims(img_clean[:,:,band_id], axis=-1)
        data_x[i, :, :, :] = img_clean

        img_ns = np.array(img_ns, dtype=np.uint8)
        img_ns = 1 - img_ns / 255.
        img_ns = cv2.resize(
            img_ns, (Params.IMG_SIZE, Params.IMG_SIZE))
        if color is not True:
            img_ns = np.expand_dims(img_ns[:,:,band_id], axis=-1)
        data_y[i, :, :, :] = img_ns

        labels.append(random_str)

        if np.mod(i, 200) == 0:
            print('generating captchas: ' + str(i) + ' / ' + str(m_samples))

    return data_x, data_y, labels



def load_data(is_training):

    DATA = {'Source1': 0, 'Source2': 0}

    if Params.task_name is 'unmixing_mnist_cifar':
        print('loading cifar10 data...')
        data_cifar10 = load_and_resize_cifar10_data(is_training=is_training)
        print('loading mnist data...')
        data_mnist = load_and_resize_mnist_data(is_training=is_training)
        DATA['Source1'] = data_cifar10
        DATA['Source2'] = data_mnist

    if Params.task_name is 'unmixing_mnist_mnist':
        print('loading mnist data...')
        data_mnist = load_and_resize_mnist_data(is_training=is_training)
        DATA['Source1'] = data_mnist
        DATA['Source2'] = data_mnist

    if Params.task_name is 'denoising':
        print('loading mnist data...')
        data_mnist = load_and_resize_mnist_data(is_training=is_training)
        DATA['Source1'] = data_mnist
        DATA['Source2'] = np.copy(data_mnist)

    if Params.task_name is 'captcha':
        print('generating captchas...')
        data_captcha_clean, data_captcha_ns, _ = load_captcha(m_samples=10000)
        DATA['Source1'] = data_captcha_clean
        DATA['Source2'] = data_captcha_ns

    print('loading data finished')

    return DATA



