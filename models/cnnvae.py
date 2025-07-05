import torch
from torch import nn, einsum
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import LayerNorm as RMSNorm  # Assuming RMSNorm is similar to LayerNorm
from einops import rearrange
from torch import optim
from math import ceil

class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(in_channels, out_channels,kernel_size, 2, kernel_size // 2)
        
        self.act = nn.SiLU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x += skip
        return self.act(self.bn2(x))



class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(in_channels // 2, eps=1e-4)
        self.conv2 = nn.Conv1d(in_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-4)

        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode='linear')
        self.act = nn.SiLU()

    def forward(self, x):
        skip = self.conv3(self.up_nn(x))
        x = self.act(self.bn1(self.conv1(self.up_nn(x))))
        x = self.conv2(x)
        x += skip
        return self.act(self.bn2(x))



class Encoder(nn.Module):
    def __init__(self, in_channels, ch, latent_dim):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, ch, 7, 1, 3)
        self.res_down1 = ResDown(ch, ch * 2)
        self.res_down2 = ResDown(ch * 2, ch * 4)
        self.res_down3 = ResDown(ch * 4, ch * 8)
        self.res_down4 = ResDown(ch * 8, ch * 16)
        self.conv_mu = nn.Conv1d(ch * 16, latent_dim,4, 1)
        self.conv_logvar = nn.Conv1d(ch * 16, latent_dim, 4, 1)
        self. act = nn.SiLU()


    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, x):
        x = self.act(self.conv_in(x))
        x = self.res_down1(x)
        x = self.res_down2(x)
        x = self.res_down3(x)
        x = self.res_down4(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)

        if self.training:
            z = self.sample(mu, logvar)
        else:
            z = mu

        return z, mu, logvar    



class Decoder(nn.Module):
    def __init__(self, out_channels, ch, latent_dim):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose1d(latent_dim, ch * 16, 4, 1)           
        self.res_up1 = ResUp(ch * 16, ch * 8)
        self.res_up2 = ResUp(ch * 8, ch * 4)
        self.res_up3 = ResUp(ch * 4, ch * 2)
        self.res_up4 = ResUp(ch * 2, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, 3, 1, 1)
        self.act = nn.SiLU()


    def forward(self, x):
        x = self.act(self.conv_t_up(x))
        x = self.res_up1(x)
        x = self.res_up2(x)
        x = self.res_up3(x)
        x = self.res_up4(x)
        x = F.tanh(self.conv_out(x))
        return x * (2 * torch.pi)







class CVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, num_labels):
        super(CVAE, self).__init__()
        self.encoder = Encoder(in_channels, 4, latent_dim)
        self.decoder = Decoder(in_channels, 4, latent_dim)
        self.cond_emb = nn.Embedding(num_labels, latent_dim * 2)

        self.proj = nn.Conv1d(latent_dim, latent_dim, 3, 1, 1)
        self.norm = nn.BatchNorm1d(latent_dim)

    def forward(self, x, cond):
        cond = self.cond_emb(cond)
        cond = rearrange(cond, 'b c -> b c 1')
        scale, shift = cond.chunk(2, dim=1)
        z, mu, logvar = self.encoder(x)
        z = z.unsqueeze(-1)
        z = self.norm(self.proj(z))
        z = z * (scale + 1) + shift
        z = F.silu(z)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, 4, latent_dim)
        self.decoder = Decoder(in_channels, 4, latent_dim)

    def encode(self, x):
        x = x.unsqueeze(1)
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def forward(self, x):
        x = x.unsqueeze(1)
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar
    




def vae_loss(recon_x, x, mu, logvar, beta=1):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kldivergence



if __name__ == '__main__':
    import os
    import re
    import seaborn as sns
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset
    from torch import optim
    import pandas as pd
    from tqdm import tqdm

    def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
        return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

       # 初始化标签和数据集
    tensor_dataset0 = torch.tensor([], dtype=torch.float32)
    labels_dataset0 = torch.tensor([], dtype=torch.int64)
    tensor_dataset1 = torch.tensor([], dtype=torch.float32)
    labels_dataset1 = torch.tensor([], dtype=torch.int64)

        # 处理dir0
    dir0 = 'data/simply2desing/Ising/uni/0/'
    
    # 第一步：先收集所有标签名称
    labels0 = []
    files = sorted(os.listdir(dir0), key=natural_sort_key)
    for file in files:
        base_name = os.path.splitext(file)[0]
        labels0.append(base_name)
    
    # 第二步：创建标签映射字典
    label_to_id0 = {label: i for i, label in enumerate(labels0)}
    id_to_label0 = {i: label for i, label in enumerate(labels0)}

    # 第三步：正式处理数据
    tensor_dataset0 = torch.tensor([], dtype=torch.float32)
    labels_dataset0 = torch.tensor([], dtype=torch.int64)
    for file in files:
        file_path = os.path.join(dir0, file)
        df = pd.read_csv(file_path)
        matrix0 = df.to_numpy()
        tensor_data = torch.tensor(matrix0, dtype=torch.float32).T
        tensor_dataset0 = torch.cat((tensor_dataset0, tensor_data), 0)
        
        # 使用预先创建好的映射字典
        base_name = os.path.splitext(file)[0]
        labels_dataset0 = torch.cat([
            labels_dataset0,
            torch.full((tensor_data.shape[0],), 
                      label_to_id0[base_name],  # 此时字典已定义
                      dtype=torch.int64)
        ])

    # 处理dir1（使用相同逻辑）
    dir1 = 'data/simply2desing/Ising/uni/1/'
    
    # 第一步：收集标签名称
    labels1 = []
    files = sorted(os.listdir(dir1), key=natural_sort_key)
    for file in files:
        base_name = os.path.splitext(file)[0]
        labels1.append(base_name)
    
    # 第二步：创建映射字典（注意ID偏移）
    label_to_id1 = {label: i + len(labels0) for i, label in enumerate(labels1)}
    
    # 第三步：处理数据
    tensor_dataset1 = torch.tensor([], dtype=torch.float32)
    labels_dataset1 = torch.tensor([], dtype=torch.int64)
    for file in files:
        file_path = os.path.join(dir1, file)
        df = pd.read_csv(file_path)
        matrix1 = df.to_numpy()
        tensor_data = torch.tensor(matrix1, dtype=torch.float32).T
        tensor_dataset1 = torch.cat((tensor_dataset1, tensor_data), 0)
        
        base_name = os.path.splitext(file)[0]
        labels_dataset1 = torch.cat([
            labels_dataset1,
            torch.full((tensor_data.shape[0],),
                      label_to_id1[base_name],  # 使用dir1的映射
                      dtype=torch.int64)
        ])
    # 合并数据集（关键修正）
    tensor_combined = torch.cat([tensor_dataset0, tensor_dataset1], dim=0)
    labels_combined = torch.cat([labels_dataset0, labels_dataset1], dim=0)
    dataset = TensorDataset(tensor_combined, labels_combined)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # 验证尺寸一致性
    print(f"特征张量尺寸: {tensor_combined.shape}")  # 应为 [总样本数, 特征维度]
    print(f"标签张量尺寸: {labels_combined.shape}")  # 应为 [总样本数]
    assert tensor_combined.shape[0] == labels_combined.shape[0], "样本数量不匹配！"

    input_dim = tensor_dataset0.shape[1]
    latent_dim = 2
    cond_dim = 8
    num_labels = len(labels0) + len(labels1)
    batch_size =512

    input_length = tensor_combined.shape[1]
    cvae = CVAE(
        in_channels=1,
        latent_dim=2,
        cond_dim=8,
        input_length=input_length
    ).to('cuda')

    optimizer = optim.Adam(cvae.parameters(), lr=1e-5)
    loss_history = []
    epochs = 1000
    with tqdm(total=epochs * len(dataloader), desc="Training Progress") as pbar:
        for epoch in range(epochs):
            cvae.train()
            train_loss = 0
            for batch_idx, (data, label) in enumerate(dataloader):
                optimizer.zero_grad()
                data = data.to('cuda')
                label = label.to('cuda')
                recon_batch, mu, logvar = cvae(data, label)
                loss = vae_loss(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"Epoch": epoch + 1, "Loss": f"{train_loss / ((batch_idx + 1) * batch_size):.4f}"})

            avg_loss = train_loss / len(dataloader.dataset)
            loss_history.append(avg_loss)

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', markersize=1.5, label='Training Loss', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('vae_loss.png')