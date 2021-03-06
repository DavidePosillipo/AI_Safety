import argparse
import os
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from classes.models_wgan import Generator, Inverter, Discriminator
from classes.searching_algorithms import iterative_search, recursive_search
from classes.dataloaders import get_mnist_dataloaders, get_fashion_mnist_dataloaders

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

def nn_classifier(x):
    predicted = le_net(x)
    _, y_hat_nn = torch.max(predicted.data, 1)
    return y_hat_nn

### Fashion MNIST
# Test data
idx = np.random.randint(10000, size = 500)
test_set_sampler = sampler.SubsetRandomSampler(idx)

all_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
test_data = datasets.FashionMNIST('./fashion_data', train=False, transform=all_transforms)
test_loader = DataLoader(test_data, batch_size=1, sampler=test_set_sampler)

searcher = recursive_search

n = len(test_loader)
output_delta_z = np.ndarray(n)

for i, data in enumerate(test_loader):
    print("test point", i, "over", n)
    x = data[0].cuda()
    y_pred = nn_classifier(x)

    output_delta_z[i] = searcher(generator,
        inverter,
        nn_classifier,
        x,
        y_pred,
        verbose = False)["delta_z"]

    print("delta_z of point", i, " :", output_delta_z[i])

np.save("test_deltaz_fashion_mnist.npy", output_delta_z)
