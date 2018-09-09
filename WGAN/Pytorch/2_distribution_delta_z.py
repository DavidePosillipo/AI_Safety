import argparse
import os
import numpy as np
import math
import sys
import cProfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from classes.models_wgan import Generator, Inverter, Discriminator
from classes.searching_algorithms import iterative_search, recursive_search
from classes.dataloaders import get_mnist_dataloaders

from LeNet import Net

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=64, dim=32)
discriminator = Discriminator(img_size=img_size, dim=32)
inverter = Inverter(img_size=img_size, latent_dim=64, dim=32)

generator.load_state_dict(torch.load("./models/gen_mnist_model_32.pt"))
inverter.load_state_dict(torch.load("./models/inv_mnist_model_32ß.pt"))

generator.cuda()
inverter.cuda()

le_net = Net()
le_net.load_state_dict(torch.load("./models/le_net.pt"))
le_net.cuda()

# Training data
dataloader, dataloader_test = get_mnist_dataloaders(batch_size=1)

def nn_classifier(x):
    predicted = le_net(x)
    _, y_hat_nn = torch.max(predicted.data, 1)
    return y_hat_nn

searcher = recursive_search

n = len(dataloader_test)

output_delta_z = np.ndarray(n)

for i, data in enumerate(dataloader_test):
    print("test point", i, "over", n)
    x = data[0].cuda()
    y = data[1].cuda()
    y_pred = nn_classifier(x)

    if y_pred != y:
        continue

    output_delta_z[i] = searcher(generator,
        inverter,
        nn_classifier,
        x,
        y,
        verbose = False)["delta_z"]

    print("delta_z of point", i, " :", output_delta_z[i])

np.save("test_deltaz.npy", output_delta_z)
