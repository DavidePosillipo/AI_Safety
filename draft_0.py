import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn

import torch.autograd as autograd
import torch

###########################################################
############### FOLDER CONFIGURATION ######################
###########################################################
os.makedirs("images", exist_ok = True)


###########################################################
############### COMMAND LINE ARGS #########################
###########################################################
# Temporary, I use an explicit dictionary for the parameters
opt = {"n_epochs": 1,
    "n_epochs_inverter": 1,
    "batch_size": 64,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_cpu": 4,
    "latent_dim": 100,
    "img_size": 28,
    "channels": 1,
    "n_critic": 5,
    "clip_value": 0.01,
    "sample_interval": 400,
    "lambda": 0.1}

img_shape = (opt["channels"], opt["img_size"], opt["img_size"])

cuda = True if torch.cuda.is_available() else False

###########################################################
############### CLASSES CONFIGURATION #####################
###########################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Linear(opt["latent_dim"], 4096),
            nn.ReLU(inplace = True)
        )

        self.model_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride = 2, padding = 1, bias = True),
            nn.ReLU(inplace = True)
        )

        self.model_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, stride = 2, padding = 3, bias = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 1, 5, stride = 2, padding = 3, output_padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model_1(z)
        #print("shape after model 1", img.shape)
        img = img.view(img.shape[0], 256, 4, 4)
        #print("shape after model 1, reshaped", img.shape)
        img = self.model_2(img)
        #print("shape after model 2, no reshape", img.shape)
        #img = img[:, :, :7, :7]
        #print("shape after model 2, reshaped", img.shape)
        img = self.model_3(img)
        #print("image generated")
        #img = img.view(img.shape[0], -1)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.model_2 = nn.Sequential(
            nn.Linear(4096, 1)
        )

    def forward(self, img):
        # Flattening the image
        #img_flat = img.view(img.shape[0], -1)
        # applying the model to the flattened image
        #print("Shape from discriminator, 1", img.shape)
        img = self.model_1(img)
        #print("Shape from discriminator, 2", img.shape)
        img = img.view(img.shape[0], 4096)
        validity = self.model_2(img)
        return validity

class Inverter(nn.Module):
    def __init__(self):
        super(Inverter, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.model_2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(1024, opt["latent_dim"])
        )

#        self.model = nn.Sequential(
#            nn.Conv2d(int(np.prod(img_shape)), 256, kernel_size = 5, stride = 2),
#            nn.LeakyReLU(0.2, inplace = True),
#            nn.LeakyReLU(0.2, inplace = True),
#            nn.Conv2d(128, 64, kernel_size = 5, stride = 2),
#            nn.LeakyReLU(0.2, inplace = True),
#            nn.Linear(64, 4096),
#            nn.LeakyReLU(0.2, inplace = True),
#            nn.Linear(4096, 1024),
#            nn.LeakyReLU(0.2, inplace = True),
#            nn.Linear(1024, opt["latent_dim"])
#        )

    def forward(self, img):
        #img_flat = img.view(img.shape[0], -1)
        img = self.model_1(img)
        img = img.view(img.shape[0], 4096)
        inverted = self.model_2(img)
        return inverted


###########################################################
######## INITIALIZATION AND DATA LOADING ##################
###########################################################
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()


# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)


dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt["batch_size"], shuffle=True)


# Setting the optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

##
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # In the original paper, alpha is the epsilon parameter
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    # The interpolates vector is denoted as x^hat in the original paper
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Applying the Discriminator to the interpolates vector
    d_interpolates = D(interpolates)

    # A temporary variable to store the result of the gradient
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    # The "input" parameter denotes the variables respect to the gradient is calculated
    # The "outputs" parameter denotes the vector of which we want the gradient
    # The "grad_outputs" parameter denotes the variable that has to store the results
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    # Computation of the gradient penalty (without lamda) as indicated in the paper
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


###########################################################
######## TRAINING OF DISCRIMINATOR AND GENERATOR ##########
###########################################################
# Loss weight for gradient penalty
lambda_gp = 10

batches_done = 0

# Loop over the number of epochs
for epoch in range(opt["n_epochs"]):
    # For each epoch, all the images are processed, using batch_size images for each sub-iteration
    # (imgs, _) is a tuple with the first element made of the batch_size images, and the second element ("_") is made of the labels of the images
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Zero the gradients before running the backward pass.
        optimizer_D.zero_grad()

        # Sample noise as generator input
        # Line 11 of the paper
        # imgs.shape[0] is given by batch_size (it's different only for the last iteration). So here a batch of latent variables z is created, each of them is a vector of size latent_dim.
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt["latent_dim"]))))
        #print("z shape", z.shape)

        # Generate a batch of images
        fake_imgs = generator(z)
        #print("shape of fake_imgs", fake_imgs.shape)

        # Real images
        real_validity = discriminator(real_imgs)
        #print("shape of real_validity", real_validity.shape)
        #print("real validity", real_validity)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        #print("fake validity", fake_validity)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        # Adversarial loss
        # Equation number 3 in the WGAN paper. The expected value is estimated using the mean (line 7 of the paper algorithm)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        # the backward() method computes gradient of the loss with respect to all the learnable parameters of
        # the model
        d_loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        # Here we are updating the parameters of the Discriminator (line 9 of the paper algorithm)
        optimizer_D.step()

        # Zero the gradients before running the backward pass.
        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        # This corresponds to the for loop of the paper, line 2-10 of the algorithm.
        # In the paper, for n_critic times only the Discriminator is trained, then after n_critic iteration, the Generator is trained once. This until convergence of the Generator parameters (theta)
        if i % opt["n_critic"] == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            # The loss function of the Generator is the mean of -D(G(z))
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt["n_epochs"],
                                                            i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

            if batches_done % opt["sample_interval"] == 0:
                save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

            batches_done += opt["n_critic"]

        #if i > 10:
        #    break

torch.save(generator.state_dict(), "generator.pt")
torch.save(discriminator.state_dict(), "discriminator.pt")

###########################################################
############### TRAINING OF INVERTER ######################
###########################################################
# Resetting the dataloader
dataloader_2 = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt["batch_size"], shuffle=True)

# Initialize the inverter
inverter = Inverter()
optimizer_I = torch.optim.Adam(inverter.parameters(), lr = opt["lr"], betas = (opt["b1"], opt["b2"]))

#batches_done = 0

if cuda:
    inverter.cuda()

for epoch in range(opt["n_epochs_inverter"]):
    for i, (imgs, _) in enumerate(dataloader_2):

        # real images
        x = Variable(imgs.type(Tensor))

        optimizer_I.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt["latent_dim"]))))

        z_prime = inverter(x)
        #print("z_prime shape", z_prime.shape)
        x_reconstructed = generator(z_prime)
        #print("x_reconstructed shape", x_reconstructed.shape)

        x_prime = generator(z)
        #print("x_prime shape", x_prime.shape)
        z_reconstructed = inverter(x_prime)
        #print("z_reconstructed shape", z_reconstructed.shape)

        #print(z_reconstructed.shape)

        # loss function of the inverter
        # (equation 2 of the paper)
        inverter_loss = ((x - x_reconstructed) ** 2).mean() + opt["lambda"] * ((z - z_reconstructed) ** 2).mean()

        inverter_loss.backward()
        optimizer_I.step()

        if i % opt["n_critic"] == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [I loss: %f]" % (epoch, opt["n_epochs"], i, len(dataloader_2), inverter_loss.item()))

        #if i>20:
        #    break

    #print(inverter)
torch.save(inverter.state_dict(), "inverter.pt")
