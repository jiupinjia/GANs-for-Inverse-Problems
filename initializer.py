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

class TrainingParamInitialization():
    """Initialize all params for training and testing"""

    def __init__(self):

        # choice a task from the following options:
        # 'unmixing_mnist_cifar'
        # 'unmixing_mnist_mnist'
        # 'denoising'
        # 'captcha'
        self.task_name = 'unmixing_mnist_cifar'

        # choice a GAN model from the following options:
        # 'Vanilla_GAN'
        # 'LS_GAN'
        # 'W_GAN'
        self.gan_model = 'Vanilla_GAN'

        self.batch_size = 8  # Batch size for training
        self.max_iters = 10000  # Maximum training iterations
        self.with_random_dual_mapping = True  # True or False
        self.with_paired_loss = False  # True or False

        self.checkpoint_dir = 'checkpoints'  # Directory to save the checkpoints
        self.sample_dir = 'samples'  # Directory to save the samples on training
        self.result_dir = 'results'  # Directory to save the generated images

        # set image size
        if self.task_name is 'unmixing_mnist_cifar':
            self.IMG_SIZE = 64  # Image size
            self.D_input_size = 64  # Input image size of the D_1 and D_2
            self.G_input_size = 64  # Output image size of G
        if self.task_name is 'unmixing_mnist_mnist':
            self.IMG_SIZE = 64  # Image size
            self.D_input_size = 64  # Input image size of the D_1 and D_2
            self.G_input_size = 64  # Output image size of G
            self.max_iters = 120000  # Need more training iters
        elif self.task_name is 'denoising':
            self.IMG_SIZE = 64  # Image size
            self.D_input_size = 64  # Input image size of the D_1 and D_2
            self.G_input_size = 64  # Output image size of G
        elif self.task_name is 'captcha':
            self.IMG_SIZE = 128  # Image size
            self.D_input_size = 128  # Input image size of the D_1 and D_2
            self.G_input_size = 128  # Output image size of G

        # set learning rate
        if self.gan_model is 'Vanilla_GAN':
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-6
        elif self.gan_model is 'LS_GAN':
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-5
        elif self.gan_model is 'W_GAN':
            self.g_learning_rate = 5e-5
            self.d_learning_rate = 5e-6

