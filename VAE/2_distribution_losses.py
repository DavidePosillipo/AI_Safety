import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from classes.models_vae import VAE

import numpy as np

# loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAE = VAE()
VAE.load_state_dict(torch.load("./models/VAE_mnist.pt"))
VAE.to(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True, **kwargs)

data_test_losses = np.ndarray(len(test_loader))

for i, data in enumerate(test_loader):
    data = data[0].to(device)
    recon, mu, logvar = VAE(data)
    data_loss = loss_function(recon, data, mu, logvar)
    data_test_losses[i] = data_loss
    #print("loss function for test input ", i, " :", data_loss)

np.save("test_data_losses.npy", data_test_losses)
print("MNIST test data loss computation completed.")
