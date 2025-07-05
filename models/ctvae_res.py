import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import BatchNorm1d as RMSNorm
from tqdm import tqdm 
from einops import rearrange
from torch import optim
from math import ceil

# class RMSNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.g = nn.Parameter(torch.ones(1, dim, 1))

#     def forward(self, x):
#         return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))   # 可学习缩放参数
        self.eps = eps  # 避免除零

    def forward(self, x):
        # 计算均方根
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        # 归一化并应用缩放参数
        return x / rms * self.g

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class VAE(nn.Module):
    def __init__(self,in_channel, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.enc_bock = nn.Sequential(
            nn.Conv1d(in_channel, 32, 3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(32, 64,  3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 128, 3, 1, padding='same'),
            nn.SiLU(),
        )

        self.attn_1 = LinearAttention(input_dim)
        self.attn_2 = Attention(128)

        self.mu_mlp = nn.Sequential(
            nn.Linear(input_dim * 128, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        ) 
        self.logvar_mlp = nn.Sequential(
            nn.Linear(input_dim * 128, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

        self.dec_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, input_dim * 128),
            nn.SiLU()
        )

        self.dec_block = nn.Sequential(
            nn.Conv1d(128, 64, 3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 32, 3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(32, in_channel, 3, 1, padding='same'),
        )

        self.shortcut = nn.Conv1d(in_channel, 128, 1) if in_channel != 128 else nn.Identity()
        self.center = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        x = x.unsqueeze(1)
        res = self.shortcut(x)
        h = self.enc_bock(x)
        h = rearrange(h, 'b c n -> b n c')
        h = self.attn_1(h)
        h = rearrange(h, 'b n c -> b c n')
        h = h - self.center
        h = h + res
        h = rearrange(h, 'b c n -> b (c n)')
        mu = F.silu(self.mu_mlp(h))
        logvar = F.silu(self.logvar_mlp(h))
        return mu, logvar
    
    def reparameterizer(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.dec_mlp(z)
        h = rearrange(h, 'b (c n) -> b c n', c=128)
        h = F.tanh(self.dec_block(h))
        h = rearrange(h, 'b c n -> b (c n)', c=1)
        h = torch.tensor(2 * torch.pi) * h
        return h
    
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterizer(mu, logvar)
        return self.decode(z), mu, logvar
    


class CVAE(nn.Module):
    def __init__(self,in_channel, input_dim, latent_dim, num_labels):
        super(CVAE, self).__init__()

        self.enc_bock = nn.Sequential(
            nn.Conv1d(in_channel, 32, 3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(32, 64,  3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 128, 3, 1, padding='same'),
            nn.SiLU(),
        )

        self.attn_1 = LinearAttention(input_dim)
        self.attn_2 = Attention(128)

        self.mu_mlp = nn.Sequential(
            nn.Linear(input_dim * 128, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        ) 
        self.logvar_mlp = nn.Sequential(
            nn.Linear(input_dim * 128, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

        self.dec_mlp = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.SiLU(),
            nn.Linear(256, input_dim * 128),
            nn.SiLU()
        )

        self.dec_block = nn.Sequential(
            nn.Conv1d(128, 64, 3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 32, 3, 1, padding='same'),
            nn.SiLU(),
            nn.Conv1d(32, in_channel, 3, 1, padding='same'),
        )

        self.shortcut = nn.Conv1d(in_channel, 128, 1) if in_channel != 128 else nn.Identity()
        self.center = nn.Parameter(torch.zeros(input_dim))

        self.embed = nn.Embedding(num_labels, 10)

    def encode(self, x):
        x = x.unsqueeze(1)
        res = self.shortcut(x)
        h = self.enc_bock(x)
        h = rearrange(h, 'b c n -> b n c')
        h = self.attn_1(h)
        h = rearrange(h, 'b n c -> b c n')
        h = h - self.center
        h = h + res
        h = rearrange(h, 'b c n -> b (c n)')
        mu = F.silu(self.mu_mlp(h))
        logvar = F.silu(self.logvar_mlp(h))
        return mu, logvar
    
    def reparameterizer(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        c = self.embed(c)
        h = torch.cat((z, c), dim=-1)
        h = self.dec_mlp(h)
        h = rearrange(h, 'b (c n) -> b c n', c=128)
        h = F.tanh(self.dec_block(h))
        h = rearrange(h, 'b c n -> b (c n)', c=1)
        h = torch.tensor(2 * torch.pi) * h
        return h
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparameterizer(mu, logvar)
        return self.decode(z, c), mu, logvar