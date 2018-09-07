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

test_loader_fashion = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../FashionData', download = True, train=False, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True, **kwargs)

data_test_fashion_losses = np.ndarray(len(data_test_fashion))
data_test_fashion = list(enumerate(test_loader_fashion))

for i in range(len(data_test_fashion)):
    data = torch.Tensor(data_test_fashion[i][1][0])
    data = data.to(device)
    recon, mu, logvar = model(data)
    data_loss = loss_function(recon, data, mu, logvar)
    data_test_fashion_losses[i] = data_loss

np.save("test_data_fashion_losses.npy", data_test_fashion_losses)
print("MNIST Fashion test data loss computation completed.")
