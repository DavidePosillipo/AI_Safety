import torch.nn as nn

###########################################################
############### CLASSES CONFIGURATION #####################
###########################################################
class Generator(nn.Module):
    def __init__(self, latent_dim, ngf, nc):
        super(Generator, self).__init__()

        self.model_1 = nn.Sequential(
           # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(    ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True) #(ngf*4) x 8 x 8
        )

        self.model_2 = nn.Sequential(
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(    ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 14 x 14
            #nn.ConvTranspose2d(    ngf * 2,     ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),

            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(     ngf * 2,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        #print("ciao")
        img = self.model_1(z)
        # Sub-setting like in the original implementation
        img = img[:, :, :7, :7]
        img = self.model_2(img)
        #print("ciao 2, shape of img generated", img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf*2, 2, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 2, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        #print("val shape 1", validity.shape)
        validity = validity.view(-1, 1).squeeze(1)
        #print("val shape 2", validity.shape)
        return validity

class Inverter(nn.Module):
    def __init__(self, latent_dim, nif, nc):
        super(Inverter, self).__init__()

        self.model_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nif, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nif) x 32 x 32
            nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nif * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nif*2) x 16 x 16
            nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nif * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nif*4) x 8 x 8
            nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nif * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model_2 = nn.Sequential(
            nn.Linear(nif * nif, nif * 16),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(nif * 16, latent_dim)
        )

    def forward(self, img):
        img = self.model_1(img)
        # Reshaping for the Linear layers
        img = img.view(img.shape[0], nif * nif)
        inverted = self.model_2(img)
        return inverted
