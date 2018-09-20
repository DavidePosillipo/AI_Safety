from LeNet import Net
import numpy as np

import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from classes.models_wgan import Generator, Inverter, Discriminator
from classes.searching_algorithms import iterative_search, recursive_search
from classes.dataloaders import get_mnist_dataloaders


def nn_classifier(x):
    predicted = le_net(x)
    _, y_hat_nn = torch.max(predicted.data, 1)
    return y_hat_nn


def load_test_data():
    idx = np.random.randint(10000, size = 500)
    test_set_sampler = sampler.SubsetRandomSampler(idx)

    all_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    test_data = datasets.MNIST('./data', train=False, transform=all_transforms)
    return DataLoader(test_data, batch_size=1, sampler=test_set_sampler)


def load_state_dict(network, path):
    """ this could be a method of the Gene/Discr/Inverter classes """
    try:
        network.load_state_dict(torch.load(path))
        network.cuda()

    except RuntimeError:
        network.load_state_dict(torch.load(path, map_location='cpu'))


def find_delta_z(test_loader):
    searcher = recursive_search

    n = len(test_loader)

    output_delta_z = np.ndarray(n)

    for i, data in enumerate(test_loader):
        print("test point", i, "over", n)
        x = data[0].cuda()
        y = data[1].cuda()
        y_pred = nn_classifier(x)

        if y_pred != y:
            continue

        output_delta_z[i] = searcher(generator,
            inverter,
            nn_classifier,
            x,
            y,
            verbose = False)["delta_z"]

        print("delta_z of point", i, " :", output_delta_z[i])

    np.save("test_deltaz.npy", output_delta_z)


if __name__ == '__main__':
    img_size = (32, 32, 1)

    generator = Generator(img_size=img_size, latent_dim=64, dim=32)
    discriminator = Discriminator(img_size=img_size, dim=32)
    inverter = Inverter(img_size=img_size, latent_dim=64, dim=32)
    le_net = Net()

    load_state_dict(generator, "./models/gen_mnist_model_32.pt")
    load_state_dict(inverter, "./models/inv_mnist_model_32.pt")
    load_state_dict(le_net, "./models/le_net.pt")

    generator.eval()
    inverter.eval()
    le_net.eval()

    test_loader = load_test_data()

    find_delta_z(test_loader)
