import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class TrainerWGANInv():
    def __init__(
            self,
            generator,
            discriminator,
            inverter,
            gp_weight=10,
            critic_iterations=5,
            lambda_inv=0.1,
            print_every=50,
            use_cuda=False
    ):

        self.generator = generator
        self.discriminator = discriminator
        self.inverter = inverter

        self.losses = {'G': [], 'D': [], 'I': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.lambda_inv = lambda_inv
        self.print_every = print_every

        if self.use_cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.inverter.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.discriminator(data)
        d_generated = self.discriminator(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.discriminator.optimizer.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.discriminator.optimizer.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _inverter_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        z = self.generator.sample_latent(batch_size)
        if self.use_cuda:
            z = z.cuda()
        #print(z)
        #print("size of z", z.size())

        x_prime = self.generator(z)

        #print("size of x_prime", x_prime.size())

        # Calculate probabilities on real and generated data
        x = Variable(data)
        if self.use_cuda:
            x = data.cuda()
        z_prime = self.inverter(x)

        x_reconstructed = self.generator(z_prime)

        z_reconstructed = self.inverter(x_prime)

        # Create total loss and optimize
        self.inverter.optimizer.zero_grad()
        i_loss = ((x - x_reconstructed) ** 2).mean() + self.lambda_inv * ((z - z_reconstructed) ** 2).mean()
        i_loss.backward()

        self.inverter.optimizer.step()

        # Record loss
        self.losses['I'].append(i_loss.item())

    def _generator_train_iteration(self, data):
        """ """
        self.generator.optimizer.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.discriminator(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.generator.optimizer.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            self._inverter_train_iteration(data[0])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("I: {}".format(self.losses['I'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.generator.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.generator(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        if save_training_gif:
            imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
                            training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.generator.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.generator(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
