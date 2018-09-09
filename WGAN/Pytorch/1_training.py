import torch
import torch.optim as optim

from classes.dataloaders import get_mnist_dataloaders
from classes.models_wgan import Generator, Discriminator, Inverter
from classes.training_wgan_inverter import TrainerWGANInv

import numpy as np

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=64, dim=32)
discriminator = Discriminator(img_size=img_size, dim=32)
inverter = Inverter(img_size=img_size, latent_dim=64, dim=32)

print(generator)
print(discriminator)
print(inverter)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
I_optimizer = optim.Adam(inverter.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = TrainerWGANInv(generator,
    discriminator,
    inverter,
    G_optimizer,
    D_optimizer,
    I_optimizer,
    use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

name = 'mnist_model_32'
torch.save(trainer.G.state_dict(), './models/gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './models/dis_' + name + '.pt')
torch.save(trainer.I.state_dict(), './models/inv_' + name + '.pt')

np.save("gdi_losses_32.npy", trainer.losses)
