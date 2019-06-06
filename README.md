# GANs for Solving Inverse Problems
The test code for: "Adversarial Training for Solving Inverse Problems".

Author: zzhengxi@umich.edu

With this project, you can train a model to solve the following inverse problems:
- Separating superimposed images on MNIST and CIFAR-10 datasets.
- Image denoising on MNIST
- Removing speckle and streak noise in CAPTCHAs

All the above tasks are trained without any help of pair-wise supervision.

## Prerequisites

- python 3.5
- anaconda>=5.2
- opencv>=3.4
- tensorflow >=1.8


## Usage

Train:

    $ python igan.py

Test:

    $ python deploy.py 

Configurations:

    $ vi initializer.py 


## Results

#### Separating superimposed images

![](results-unmixing.jpg)

#### Breaking CAPTCHAs

![](results-captcha.jpg)

#### Denoising

![](results-denoising.jpg)

