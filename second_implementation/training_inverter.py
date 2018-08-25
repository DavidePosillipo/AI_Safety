import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer_inverter():
    def __init__(self, inverter, generator, inv_optimizer,
                 lambda_inv=0.1,  print_every=50,
                 use_cuda=False):
        self.I = inverter
        self.I_opt = inv_optimizer
        self.G = generator
        self.losses = {'I': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.print_every = print_every
        self.lambda_inv = lambda_inv

        if self.use_cuda:
            self.I.cuda()
            self.G.cuda()

    def _inverter_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        z = self.G.sample_latent(batch_size)
        if self.use_cuda:
            z = z.cuda()
        #print(z)
        #print("size of z", z.size())

        x_prime = self.G(z)

        #print("size of x_prime", x_prime.size())

        # Calculate probabilities on real and generated data
        x = Variable(data)
        if self.use_cuda:
            x = data.cuda()
        z_prime = self.I(x)

        x_reconstructed = self.G(z_prime)

        z_reconstructed = self.I(x_prime)

        # Create total loss and optimize
        self.I_opt.zero_grad()
        i_loss = ((x - x_reconstructed) ** 2).mean() + self.lambda_inv * ((z - z_reconstructed) ** 2).mean()
        i_loss.backward()

        self.I_opt.step()

        # Record loss
        self.losses['I'].append(i_loss.item())



    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._inverter_train_iteration(data[0])

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("I: {}".format(self.losses['I'][-1]))

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)


    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
