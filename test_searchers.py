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

from second_implementation.models import Generator, Inverter, Discriminator
from searching_algorithms import iterative_search, recursive_search

from sklearn.metrics import accuracy_score
import pickle

img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)
inverter = Inverter(img_size=img_size, latent_dim=100, dim=16)

generator.load_state_dict(torch.load("second_implementation/gen_mnist_model.pt", map_location='cpu'))
inverter.load_state_dict(torch.load("second_implementation/inv_mnist_model.pt", map_location='cpu'))


#rf_pretrained = pickle.load(open("mnist_rf_9045.sav", 'rb'), encoding = "latin1")

#print(rf_pretrained)

# Training data
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(32),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=1, shuffle=True)

data = list(enumerate(dataloader))

data[0][1][0].shape

X = []
y = []
for i in data:
    X.append(i[1][0].numpy())
    y.append(i[1][1].numpy())

X_2 = np.array(X)
X_2 = np.reshape(X_2, (len(dataloader), 1024))
type(X_2)

y_2 = np.array(y).reshape(len(dataloader), )

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(max_depth = 10, random_state = 0)

clf.fit(X_2, y_2)


dataloader_test = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(32),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=1, shuffle=True)

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

predicted[0]
y_test[0]

print("accuracy of the black box classifier", accuracy)

def rf_classifier(x):
    return clf.predict(np.reshape(x, (-1, 1024)))

searcher = iterative_search

n = len(data_test)
output_delta_z = np.ndarray(n)
for i in np.arange(n):
    print("test point", i, "over", n)
    #print(inverter(data_test[i][1][0]))
    #print("here")
    output_delta_z[i] = searcher(generator,
        inverter,
        rf_classifier,
        data_test[i][1][0],
        data_test[i][1][1],
        verbose = False)["delta_z"]
    print("delta_z of point", i, " :", output_delta_z[i])

np.save("test_deltaz.npy", output_delta_z)
