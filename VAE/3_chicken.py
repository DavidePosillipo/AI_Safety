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

ck = io.imread('chicken.jpg', as_gray=True)
ck = resize(ck, (28, 28), anti_aliasing = True)
ck = np.float32(ck)

ck_data = ck.reshape(1, 28*28)

ck = torch.Tensor(ck)
ck = ck.to(device)

recon_ck, mu_ck, logvar_ck = VAE(ck)

loss_ck = trainer_VAE.loss_function(recon_ck, ck, mu_ck, logvar_ck)

print("chicken loss", loss_ck)
