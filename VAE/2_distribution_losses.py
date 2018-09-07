import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from classes.models_vae import VAE
from classes.training_VAE import Trainer_VAE

import numpy as np

VAE = VAE()
VAE.load_state_dict(torch.load("VAE_mnist.pt"))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True, **kwargs)

data_test = list(enumerate(test_loader))
data_test_losses = np.ndarray(len(data_test))

for i in range(len(data_test)):
    data = torch.Tensor(data_test[i][1][0])
    data = data.to(device)
    recon, mu, logvar = VAE(data)
    data_loss = trainer_VAE.loss_function(recon, data, mu, logvar)
    data_test_losses[i] = data_loss
    #print("loss function for test input ", i, " :", data_loss)

np.save("test_data_losses.npy", data_test_losses)
print("MNIST test data loss computation completed.")
