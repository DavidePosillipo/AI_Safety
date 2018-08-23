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
######## TRAINING OF DISCRIMINATOR AND GENERATOR ##########
###########################################################
def train_generator_discriminator(generator, discriminator, dataloader, n_epochs, latent_dim, sample_interval, n_critic, lambda_gp, lr, b1, b2, cuda):

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ###############################################################################
    ###### Function for the computation of the gradient penalty ###################
    ###############################################################################
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
    ##################################################################################

    # Setting the optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    batches_done = 0

    # Loop over the number of epochs
    for epoch in range(n_epochs):
        # For each epoch, all the images are processed, using batch_size images for each sub-iteration
        # (imgs, _) is a tuple with the first element made of the batch_size images, and the second element ("_") is made of the labels of the images
        for i, (imgs, _) in enumerate(dataloader):
            #print("obs", i)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            #print("shape real imgs", real_imgs.shape)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Zero the gradients before running the backward pass.
            optimizer_D.zero_grad()

            # Sample noise as generator input
            # Line 11 of the paper
            # imgs.shape[0] is given by batch_size (it's different only for the last iteration). So here a batch of latent variables z is created, each of them is a vector of size latent_dim.
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim, 1, 1))))
            #print("shape of z", z.shape)
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
            if i % n_critic == 0:

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

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs,
                                                                i, len(dataloader),
                                                                d_loss.item(), g_loss.item()))

                if batches_done % sample_interval == 0:
                    save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

                batches_done += n_critic

    return generator, discriminator

            #if i > 10:
            #    break



###########################################################
############### TRAINING OF INVERTER ######################
###########################################################
def train_inverter(inverter, generator, dataloader, latent_dim, lambda_inv, n_epochs_inverter, lr, b1, b2, cuda):

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer_I = torch.optim.Adam(inverter.parameters(), lr = lr, betas = (b1, b2))

    for epoch in range(n_epochs_inverter):
        for i, (imgs, _) in enumerate(dataloader):

            # real images
            x = Variable(imgs.type(Tensor))

            optimizer_I.zero_grad()

            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim, 1, 1))))
            print("shape of z", z.shape)

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
            inverter_loss = ((x - x_reconstructed) ** 2).mean() + lambda_inv * ((z - z_reconstructed) ** 2).mean()

            inverter_loss.backward()
            optimizer_I.step()

            if i % opt.n_critic == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [I loss: %f]" % (epoch, n_epochs_inverter, i, len(dataloader), inverter_loss.item()))

            #if i>20:
            #    break

    return inverter
