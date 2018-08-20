import torch.nn as nn

###########################################################
############### CLASSES CONFIGURATION #####################
###########################################################
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(inplace = True)
        )

        self.model_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride = 2, padding = 1, bias = True), #9x9
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, 5, stride = 2, padding = 3, bias = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 1, 5, stride = 2, padding = 3, output_padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model_1(z)
        #print("shape after model 1", img.shape)
        # Reshaping the image for the Convolutional layers, obtaing a 4x4
        img = img.view(img.shape[0], 256, 4, 4)
        #print("shape after model 1, reshaped", img.shape)
        img = self.model_2(img) # 28x28
        #print("shape after model 2, no reshape", img.shape)
        #img = img[:, :, :7, :7]
        #print("shape after model 2, reshaped", img.shape)
        #img = self.model_3(img) # 28x28
        #print("image generated, shape", img.shape)
        #img = img.view(img.shape[0], -1)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.model_2 = nn.Sequential(
            nn.Linear(4096, 1)
        )

    def forward(self, img):
        # Flattening the image
        #img_flat = img.view(img.shape[0], -1)
        # applying the model to the flattened image
        #print("Shape from discriminator, 1", img.shape)
        img = self.model_1(img)
        #print("Shape from discriminator, 2", img.shape)
        # Reshaping the image for the Linear layer
        img = img.view(img.shape[0], 4096)
        validity = self.model_2(img)
        return validity

class Inverter(nn.Module):
    def __init__(self, latent_dim):
        super(Inverter, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, kernel_size = 1, stride = 2),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.model_2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, img):
        img = self.model_1(img)
        # Reshaping for the Linear layers
        img = img.view(img.shape[0], 4096)
        inverted = self.model_2(img)
        return inverted
