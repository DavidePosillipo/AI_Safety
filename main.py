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

from model_classes import Generator, Inverter, Discriminator
from searching_algorithms import iterative_search, recursive_search
from training_inverter import train_generator_discriminator, train_inverter

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--n_epochs_inverter', type=int, default=1, help='number of epochs of training for the inverter')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lambda_inv', type=float, default=0.1, help='lambda parameter for the inverter loss function')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--lambda_gp', type=int, default=10, help='loss weight for gradient penalty of WGAN')
parser.add_argument('--iterative_search', action='store_true', default=False, help='use iterative_search for the iterative search algorithm; the default is the recursive search')
parser.add_argument('--load_gen_inv', action='store_true', default=False, help='should I load already pretrained generator and inverter? Default: train from scratch')
parser.add_argument('--save_models', action='store_true', default=False, help='should I save the trained models? Default: dont save the model')

opt = parser.parse_args()

print(opt)

cuda = True if torch.cuda.is_available() else False

# img_shape = (opt.channels, opt.img_size, opt.img_size)

#######################################
####### LOADING OF DATA MNIST #########
#######################################
os.makedirs('data/mnist', exist_ok=True)

os.makedirs("images", exist_ok = True)



dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

dataloader_2 = torch.utils.data.DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

#######################################
## LOADING OF GENERATOR AND INVERTER ##
######## If already trained ###########
if opt.load_gen_inv:
    generator = Generator(latent_dim = opt.latent_dim)
    generator.load_state_dict(torch.load("generator.pt"))

    inverter = Inverter(latent_dim = opt.latent_dim)
    inverter.load_state_dict(torch.load("inverter.pt"))

else:
    print("Generator and Inverter will be trained from scratch")
    # Initialize generator and discriminator
    generator = Generator(latent_dim = opt.latent_dim)
    discriminator = Discriminator()
    inverter = Inverter(latent_dim = opt.latent_dim)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        inverter.cuda()

    generator, discriminator = train_generator_discriminator(generator, discriminator, dataloader, opt.n_epochs, opt.latent_dim, opt.sample_interval, opt.n_critic, opt.lambda_gp, opt.lr, opt.b1, opt.b2, cuda)

    inverter = train_inverter(inverter, generator, dataloader_2, opt.latent_dim, opt.lambda_inv, opt.n_epochs_inverter, opt.lr, opt.b1, opt.b2, cuda)

    if opt.save_models:
        torch.save(generator.state_dict(), "generator.pt")
        torch.save(discriminator.state_dict(), "discriminator.pt")
        torch.save(inverter.state_dict(), "inverter.pt")

#######################################
## first black box classifier #########
########### DUMMY EXAMPLE #############
######################################
import pickle
rf_pretrained = pickle.load(open("mnist_rf_9045.sav", 'rb'), encoding = "latin1")

print(rf_pretrained)

# # Training data
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data/mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=1, shuffle=True)
#
# data = list(enumerate(dataloader))
#
# data[0][1][0].shape
#
# X = []
# y = []
# for i in data:
#     X.append(i[1][0].numpy())
#     y.append(i[1][1].numpy())
#
# X_2 = np.array(X)
# X_2 = np.reshape(X_2, (len(dataloader), 784))
# type(X_2)
#
# y_2 = np.array(y).reshape(len(dataloader), )
#
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#
# clf = RandomForestClassifier(max_depth = 3, random_state = 0)
#
# clf.fit(X_2, y_2)
#
# Test data
dataloader_test = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=False, download=True,
                   transform=transforms.Compose([
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
X_test = np.reshape(X_test, (len(dataloader_test), 784))
type(X_test)

y_test = np.array(y_test).reshape(len(dataloader_test), )

predicted = rf_pretrained.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print("accuracy of the black box classifier", accuracy)

#######################################################

# Function to pass preditions to the Inverter
def rf_classifier(x):
    return rf_pretrained.predict(np.reshape(x, (-1, 784)))

# Testing the iterative search algorithm on a single input observation
# test_adversial_examples = iterative_search(generator,
#     inverter,
#     rf_classifier,
#     data[0][1][0],
#     data[0][1][1],
#     verbose = True)

if opt.iterative_search:
    searcher = iterative_search
else:
    searcher = recursive_search

#cProfile.run('searcher(generator, inverter, rf_classifier, data_test[0][1][0], data_test[0][1][1], verbose = False)')
n = 10
output_delta_z = np.ndarray(n)
for i in np.arange(n):
    print("test point", i, "over", n)
    output_delta_z[i] = searcher(generator,
        inverter,
        rf_classifier,
        data_test[i][1][0],
        data_test[i][1][1],
        verbose = False)["delta_z"]
    print("delta_z of point", i, " :", output_delta_z[i])

np.save("test_deltaz.npy", output_delta_z)

#print("delta_z", test_adversial_examples["delta_z"])
