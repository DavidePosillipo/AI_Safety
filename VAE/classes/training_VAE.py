import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class TrainerVAE():
    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


    def train(self, epoch, train_loader, log_interval):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


    def test(self, epoch, test_loader, batch_size):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'models/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


    def train_VAE(self, train_loader, test_loader, n_epochs, log_interval, batch_size):
        for epoch in range(1, n_epochs + 1):
            self.train(epoch, train_loader, log_interval)
            self.test(epoch, test_loader, batch_size)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(self.device)
                sample = self.model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),
                           'models/sample_' + str(epoch) + '.png')
