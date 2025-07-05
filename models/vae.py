import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim, num_labels):
        super(CVAE, self).__init__()

        self.encoder_fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim + cond_dim, 512)
        self.decoder_fc3 = nn.Linear(512, input_dim)

        self.embed = nn.Embedding(num_labels, cond_dim)

    def encode(self, x):
        h = self.encoder_fc1(x)
        h = F.silu(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparamterizer(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        c = self.embed(c)
        h = torch.cat((z, c), dim=-1)
        h = F.silu(self.decoder_fc1(h))
        h = F.tanh(self.decoder_fc3(h))
        h = torch.tensor(2 * torch.pi) * h
        return h
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparamterizer(mu, logvar)
        return self.decode(z, c), mu, logvar

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder_fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, 512)
        self.decoder_fc3 = nn.Linear(512, input_dim)


    def encode(self, x):
        h = self.encoder_fc1(x)
        h = F.silu(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparamterizer(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = z
        h = F.silu(self.decoder_fc1(h))
        h = F.tanh(self.decoder_fc3(h))
        h = torch.tensor(2 * torch.pi) * h
        return h
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparamterizer(mu, logvar)
        z_l2 = torch.norm(z, dim=1)
        return self.decode(z), mu, logvar, z_l2




def vae_loss(recon_x, x, mu, logvar, beta=1):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


def cvae_loss(recon_x, x, mu, logvar, beta=1):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD