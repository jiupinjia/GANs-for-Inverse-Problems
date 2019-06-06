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

        # choice a GAN model ('W_GAN', 'LS_GAN', 'Vanilla_GAN')
        self.gan_model = 'W_GAN'

        self.batch_size = 8  # Batch size for training
        self.max_iters = 50000  # Maximum training iterations
        self.with_random_dual_mapping = True  # True or false

        self.checkpoint_dir = 'checkpoints'  # Directory name to save the checkpoints
        self.sample_dir = 'samples'  # Directory name to save the samples on training
        self.result_dir = 'results'  # Directory name to save the generated images


        # defualt image size
        if self.task_name is 'unmixing':
            self.IMG_SIZE = 64  # Image size
            self.D_input_size = 64  # Input image size of the D_1 and D_2
            self.G_input_size = 64  # Output image size of G
        elif self.task_name is 'denoising':
            self.IMG_SIZE = 32  # Image size
            self.D_input_size = 32  # Input image size of the D_1 and D_2
            self.G_input_size = 32  # Output image size of G
        elif self.task_name is 'captcha':
            self.IMG_SIZE = 128  # Image size
            self.D_input_size = 128  # Input image size of the D_1 and D_2
            self.G_input_size = 128  # Output image size of G


        # Optimizer initialization
        if self.gan_model is 'W_GAN':
            self.d_clip = 0.01
            self.d_thresh = -5.
            self.optimizer = 'RMSProp'
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-5
        elif self.gan_model is 'LS_GAN':
            self.d_clip = 1e9
            self.d_thresh = 0.1
            self.optimizer = 'Adam'
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-5
        elif self.gan_model is 'Vanilla_GAN':
            self.d_clip = 1e9
            self.d_thresh = 0.5
            self.optimizer = 'Adam'
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-6

