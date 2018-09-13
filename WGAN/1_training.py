import numpy as np
import torch

from classes.dataloaders import get_mnist_dataloaders
from classes.models_wgan import Generator, Discriminator, Inverter
from classes.training_wgan_inverter import TrainerWGANInv


if __name__ == '__main__':

    data_loader, _ = get_mnist_dataloaders(batch_size=64)
    img_size = (32, 32, 1)

    optimizer_params = {
        'lr': 1e-4, 'betas': (.9, .99)
    }

    generator = Generator(
        img_size=img_size, latent_dim=64, dim=32, **optimizer_params
    )

    discriminator = Discriminator(
        img_size=img_size, dim=32, **optimizer_params
    )

    inverter = Inverter(
        img_size=img_size, latent_dim=64, dim=32, **optimizer_params
    )

    epochs = 200
    trainer = TrainerWGANInv(
        generator,
        discriminator,
        inverter,
        use_cuda=torch.cuda.is_available()
    )

    trainer.train(data_loader, epochs, save_training_gif=True)

    name = 'mnist_model_32'
    torch.save(trainer.G.state_dict(), './models/gen_' + name + '.pt')
    torch.save(trainer.D.state_dict(), './models/dis_' + name + '.pt')
    torch.save(trainer.I.state_dict(), './models/inv_' + name + '.pt')

    np.save("data/gdi_losses_32.npy", trainer.losses)
