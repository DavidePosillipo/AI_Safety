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

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=64, dim=32)
discriminator = Discriminator(img_size=img_size, dim=32)
inverter = Inverter(img_size=img_size, latent_dim=64, dim=32)

generator.load_state_dict(torch.load("./models/gen_mnist_model_32.pt"))
inverter.load_state_dict(torch.load("./models/inv_mnist_model_32.pt"))

generator.cuda()
inverter.cuda()

le_net = Net()
le_net.load_state_dict(torch.load("./models/le_net.pt"))
le_net.cuda()

# Training data
dataloader, dataloader_test = get_mnist_dataloaders(batch_size=1)
#
# data = list(enumerate(dataloader))
#
# X = []
# y = []
# for i in data:
#     X.append(i[1][0].numpy())
#     y.append(i[1][1].numpy())
#
# X_2 = np.array(X)
# X_2 = np.reshape(X_2, (len(dataloader), 1024))
# type(X_2)
#
# y_2 = np.array(y).reshape(len(dataloader), )
#
# clf = RandomForestClassifier(max_depth = 10, random_state = 0)
#
# clf.fit(X_2, y_2)
#
# data_test = list(enumerate(dataloader_test))
#
# X_test = []
# y_test = []
# for i in data_test:
#     X_test.append(i[1][0].numpy())
#     y_test.append(i[1][1].numpy())
#
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (len(dataloader_test), 1024))
#
# y_test = np.array(y_test).reshape(len(dataloader_test), )
#
# predicted = clf.predict(X_test)
# accuracy = accuracy_score(y_test, predicted)
#
# print("accuracy of the black box classifier", accuracy)
#
# def rf_classifier(x):
#     return clf.predict(np.reshape(x, (-1, 1024)))

def nn_classifier(x):
    return le_net(x)

### Chicken example
ck = io.imread('chicken.jpg', as_gray=True)
ck = resize(ck, (32, 32), anti_aliasing = True)
ck = np.float32(ck)
# rescaling to white over black
ck = 1-ck

ck_data = ck.reshape(1, 32*32)

# RF prediction
# y_hat_rf = rf_classifier(ck)
# print("estimate for the chicken (random forest)", y_hat_rf)
# ck_probabilities_rf = clf.predict_proba(ck_data)
# print("estimated probabilities for the chicken (random forest)", ck_probabilities_rf)

# NN prediction
ck_probabilities_nn = nn_classifier(torch.Tensor(ck).view(1, 1, 32, 32).cuda())
print("estimated probabilities for the chicken (LeNet)", ck_probabilities_nn)
_, y_hat_nn = torch.max(ck_probabilities_nn.data, 1)
print("estimate for the chicken (LeNet)", y_hat_nn)

# Delta_z for the chicken
searcher = recursive_search

#adversary_ck_rf = recursive_search(generator, inverter, rf_classifier, ck_data, y_hat_rf,
#                   nsamples=5000, step=0.01, verbose=False)

#print("delta_z for the chicken (random forest):", adversary_ck_rf["delta_z"])

adversary_ck_nn = recursive_search(generator, inverter, nn_classifier, torch.Tensor(ck).view(1, 1, 32, 32).cuda(), y_hat_nn,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the chicken (LeNet):", adversary_ck_nn["delta_z"])
