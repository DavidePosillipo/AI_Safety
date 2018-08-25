import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from models import Generator, Discriminator, Inverter
from training import Trainer
from training_inverter import Trainer_inverter

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)
inverter = Inverter(img_size=img_size, latent_dim=100, dim=16)

print(generator)
print(discriminator)
print(inverter)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
I_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 50
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')

trainer_inverter = Trainer_inverter(inverter, generator, I_optimizer, use_cuda=torch.cuda.is_available())
trainer_inverter.train(data_loader, epochs)

torch.save(trainer.I.state_dict(), './inv_' + name + '.pt')