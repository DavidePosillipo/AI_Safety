import argparse
import os
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from classes.models_wgan import Generator, Inverter, Discriminator
from classes.searching_algorithms import iterative_search, recursive_search
from classes.dataloaders import get_mnist_dataloaders

from LeNet import Net

img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=64, dim=32)
discriminator = Discriminator(img_size=img_size, dim=32)
inverter = Inverter(img_size=img_size, latent_dim=64, dim=32)

generator.load_state_dict(torch.load("./models/gen_mnist_model_32.pt"))
inverter.load_state_dict(torch.load("./models/inv_mnist_model_32.pt"))

generator.cuda()
generator.eval()
inverter.cuda()
inverter.eval()

le_net = Net()
le_net.load_state_dict(torch.load("./models/le_net.pt"))
le_net.eval()
le_net.cuda()

searcher = recursive_search

# Training data
dataloader, dataloader_test = get_mnist_dataloaders(batch_size=1)


def nn_classifier(x):
    predicted = le_net(x)
    _, y_hat_nn = torch.max(predicted.data, 1)
    return y_hat_nn

### Chicken example
ck = io.imread('chicken.jpg', as_gray=True)
ck = resize(ck, (32, 32), anti_aliasing = True)
ck = np.float32(ck)
# rescaling to white over black
ck = 1-ck
ck_tensor = torch.Tensor(ck).view(1, 1, 32, 32).cuda()

# NN prediction
ck_probabilities_nn = torch.exp(le_net(ck_tensor))
print("estimated probabilities for the chicken (LeNet)", ck_probabilities_nn)
y_hat_nn = nn_classifier(ck_tensor)
print("estimate for the chicken (LeNet)", y_hat_nn)

# Delta_z for the chicken
adversary_ck_nn = recursive_search(generator, inverter, nn_classifier, ck_tensor, y_hat_nn,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the chicken (LeNet):", adversary_ck_nn["delta_z"])

### Let's try with a falafel picture
fl = io.imread('falafel.jpg', as_gray=True)
fl = resize(fl, (32, 32), anti_aliasing = True)
fl = np.float32(fl)
fl = 1-fl
fl_tensor = torch.Tensor(fl).view(1, 1, 32, 32).cuda()

# NN prediction
fl_probabilities_nn = torch.exp(le_net(fl_tensor))
print("estimated probabilities for the falafel (LeNet)", fl_probabilities_nn)
y_hat_nn = nn_classifier(fl_tensor)
print("estimate for the falafel (LeNet)", y_hat_nn)

# Delta_z for the falafel
adversary_fl_nn = recursive_search(generator, inverter, nn_classifier, fl_tensor, y_hat_nn,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the falafel (LeNet):", adversary_fl_nn["delta_z"])
