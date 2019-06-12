"""
The test code for:
"Adversarial Training for Solving Inverse Problems"
using Tensorflow.

Author: Zhengxia Zou (zzhengxi@umich.edu)

With this project, you can train a model to solve the following
inverse problems:
- on MNIST and CIFAR-10 datasets for separating superimposed images.
- image denoising on MNIST
- remove speckle and streak noise in CAPTCHAs
All the above tasks are trained without any help of pair-wise supervision.

Jun., 2019
"""

class TrainingParamInitialization():
    """Initialize all params for training and testing"""

    def __init__(self):

        # choice a task ('unmixing', 'denoising', 'captcha')
        self.task_name = 'captcha'

        self.batch_size = 16  # Batch size for training
        self.max_iters = 50000  # Maximum training iterations
        self.with_random_dual_mapping = True  # True or false

        self.checkpoint_dir = 'checkpoints'  # Directory name to save the checkpoints
        self.sample_dir = 'samples'  # Directory name to save the samples on training
        self.result_dir = 'results'  # Directory name to save the generated images


        if self.task_name is 'unmixing':
            # choice a GAN model ('Vanilla_GAN', 'LS_GAN', 'W_GAN')
            self.gan_model = 'Vanilla_GAN'
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-6
            # self.gan_model = 'LS_GAN'
            # self.g_learning_rate = 5e-5
            # self.d_learning_rate = 5e-5
            # self.gan_model = 'W_GAN'
            # self.g_learning_rate = 5e-5
            # self.d_learning_rate = 5e-5
            self.IMG_SIZE = 64  # Image size
            self.D_input_size = 64  # Input image size of the D_1 and D_2
            self.G_input_size = 64  # Output image size of G

        elif self.task_name is 'denoising':
            # choice a GAN model ('Vanilla_GAN', 'LS_GAN', 'W_GAN')
            self.gan_model = 'Vanilla_GAN'
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-6
            # self.gan_model = 'LS_GAN'
            # self.g_learning_rate = 5e-5
            # self.d_learning_rate = 5e-5
            # self.gan_model = 'W_GAN'
            # self.g_learning_rate = 5e-5
            # self.d_learning_rate = 5e-5
            self.IMG_SIZE = 64  # Image size
            self.D_input_size = 64  # Input image size of the D_1 and D_2
            self.G_input_size = 64  # Output image size of G

        elif self.task_name is 'captcha':
            # choice a GAN model ('Vanilla_GAN', 'LS_GAN', 'W_GAN')
            self.gan_model = 'Vanilla_GAN'
            self.g_learning_rate = 1e-5
            self.d_learning_rate = 1e-6
            # self.gan_model = 'LS_GAN'
            # self.g_learning_rate = 1e-5
            # self.d_learning_rate = 1e-5
            # self.gan_model = 'W_GAN'
            # self.g_learning_rate = 1e-5
            # self.d_learning_rate = 1e-6
            self.IMG_SIZE = 128  # Image size
            self.D_input_size = 128  # Input image size of the D_1 and D_2
            self.G_input_size = 128  # Output image size of G
