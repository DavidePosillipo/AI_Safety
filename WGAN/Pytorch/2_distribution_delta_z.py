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

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=64, dim=64)
discriminator = Discriminator(img_size=img_size, dim=64)
inverter = Inverter(img_size=img_size, latent_dim=64, dim=64)

generator.load_state_dict(torch.load("./models/gen_mnist_model_64.pt"))
inverter.load_state_dict(torch.load("./models/inv_mnist_model_64.pt"))

generator.cuda()
inverter.cuda()

# Training data
dataloader, dataloader_test = get_mnist_dataloaders(batch_size=1)

data = list(enumerate(dataloader))

X = []
y = []
for i in data:
    X.append(i[1][0].numpy())
    y.append(i[1][1].numpy())

X_2 = np.array(X)
X_2 = np.reshape(X_2, (len(dataloader), 1024))
type(X_2)

y_2 = np.array(y).reshape(len(dataloader), )

clf = RandomForestClassifier(max_depth = 10, random_state = 0)

clf.fit(X_2, y_2)

data_test = list(enumerate(dataloader_test))

X_test = []
y_test = []
for i in data_test:
    X_test.append(i[1][0].numpy())
    y_test.append(i[1][1].numpy())

X_test = np.array(X_test)
X_test = np.reshape(X_test, (len(dataloader_test), 1024))

y_test = np.array(y_test).reshape(len(dataloader_test), )

predicted = clf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print("accuracy of the black box classifier", accuracy)

def rf_classifier(x):
    return clf.predict(np.reshape(x, (-1, 1024)))

searcher = recursive_search

n = len(data_test)
output_delta_z = np.ndarray(n)
for i in np.arange(n):
    print("test point", i, "over", n)
    x = data_test[i][1][0]
    y = data_test[i][1][1]
    y_pred = rf_classifier(x)
    if y_pred != y:
        continue
    output_delta_z[i] = searcher(generator,
        inverter,
        rf_classifier,
        x.cuda(),
        y,
        verbose = False)["delta_z"]
    print("delta_z of point", i, " :", output_delta_z[i])

np.save("test_deltaz.npy", output_delta_z)
