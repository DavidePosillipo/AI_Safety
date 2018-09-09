import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from classes.models_vae import VAE

import numpy as np

import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

VAE = VAE()
VAE.load_state_dict(torch.load("./models/VAE_mnist.pt"))
VAE.to(device)

### Chicken
ck = io.imread('chicken.jpg', as_gray=True)
ck = resize(ck, (28, 28), anti_aliasing = True)
ck = 1 - ck
ck = np.float32(ck)

ck = torch.Tensor(ck)
ck = ck.to(device)

recon_ck, mu_ck, logvar_ck = VAE(ck)

loss_ck = loss_function(recon_ck, ck, mu_ck, logvar_ck)

print("chicken loss", loss_ck)


### Falafel
fl = io.imread('falafel.jpg', as_gray=True)
fl = resize(fl, (28, 28), anti_aliasing = True)
fl = 1 - fl
fl = np.float32(fl)

fl = torch.Tensor(fl)
fl = fl.to(device)

recon_fl, mu_fl, logvar_fl = VAE(fl)

loss_fl = loss_function(recon_fl, fl, mu_fl, logvar_fl)

print("falafel loss", loss_fl)
