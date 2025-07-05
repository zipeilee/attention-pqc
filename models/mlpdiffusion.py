import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim=64, num_classes=2, t_embed_dim=128, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.t_embed_dim = t_embed_dim
        
        # 标签嵌入层
        self.label_embed = nn.Embedding(num_classes, t_embed_dim)
        
        # 时间步投影网络
        self.time_embed = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )
        
        # 共享的MLP网络
        self.shared_mlp = nn.ModuleList()
        for _ in range(num_layers):
            self.shared_mlp.append(nn.Sequential(
                nn.Linear(t_embed_dim, t_embed_dim),
                nn.SiLU(),
            ))
        
        # 条件调制投影
        self.condition_blocks = nn.ModuleList()
        for _ in range(num_layers + 1):  # +1 for input projection
            self.condition_blocks.append(nn.Sequential(
                nn.Linear(t_embed_dim, t_embed_dim * 2),
                nn.SiLU(),
                nn.Linear(t_embed_dim * 2, t_embed_dim * 2),
            ))
        
        # 输入输出投影
        self.input_proj = nn.Linear(input_dim, t_embed_dim)
        self.output_proj = nn.Linear(t_embed_dim, input_dim)
        
    def forward(self, x, labels, t):
        # 时间步嵌入
        t_embed = timestep_embedding(t, self.t_embed_dim)
        t_embed = self.time_embed(t_embed)
        
        # 标签嵌入
        label_embed = self.label_embed(labels)
        
        # 合并条件
        condition = t_embed + label_embed
        
        # 输入投影
        h = self.input_proj(x)
        
        # 通过各层处理
        for i, layer in enumerate(self.shared_mlp):
            # 应用条件调制
            mod = self.condition_blocks[i](condition)
            scale, shift = mod.chunk(2, dim=-1)
            h = h * (1 + scale) + shift
            
            # 通过MLP层
            h = layer(h)
        
        # 最终调制
        mod = self.condition_blocks[-1](condition)
        scale, shift = mod.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        
        return self.output_proj(h)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionWrapper:
    def __init__(self, model, beta_schedule='linear', T=1000,
                 beta_start=1e-4, beta_end=2e-2):
        self.model = model
        self.T = T
        
        # 注册beta调度参数
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, T)
        elif beta_schedule == 'cosine':
            raise NotImplementedError
        else:
            raise ValueError
            
        alphas = 1. - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1. - alpha_bar))
    
    def register_buffer(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.to(next(self.model.parameters()).device)
        setattr(self, name, value)
        
    def train_step(self, x0, labels, optimizer):
        optimizer.zero_grad()
        
        device = x0.device
        b = x0.size(0)
        
        # 随机采样时间步
        t = torch.randint(0, self.T, (b,), device=device)
        
        # 采样噪声
        eps = torch.randn_like(x0)
        
        # 加噪过程
        sqrt_alpha = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * eps
        
        # 预测噪声
        eps_pred = self.model(xt, labels, t)
        
        # 计算损失
        loss = F.mse_loss(eps_pred, eps)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def sample(self, labels, sample_shape):
        device = next(self.model.parameters()).device
        num_samples = labels.size(0)
        
        self.model.eval()
        with torch.no_grad():
            # 初始化随机噪声
            x_t = torch.randn((num_samples, sample_shape), device=device)
            
            # 反向扩散过程
            for t in reversed(range(self.T)):
                # 准备时间步t的张量
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                
                # 预测噪声
                eps_pred = self.model(x_t, labels, t_batch)
                
                # 计算alpha相关参数
                alpha_bar_t = self.alpha_bar[t]
                alpha_bar_prev = self.alpha_bar[t-1] if t > 0 else 1.0
                alpha_t = alpha_bar_t / alpha_bar_prev
                beta_t = self.betas[t]
                
                # 计算均值
                mu = (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t)
                
                # 添加噪声（最后一步不添加）
                if t > 0:
                    sigma_t = torch.sqrt(beta_t)
                    z = torch.randn_like(x_t)
                    x_t = mu + sigma_t * z
                else:
                    x_t = mu
        
        self.model.train()
        return x_t

def conditional_diffusion_sample(
    model,                # 训练好的模型
    labels,               # 条件标签 tensor (num_samples,)
    sample_shape,         # 样本形状（如64）
    alpha_bar,            # 从DiffusionWrapper获取的alpha_bar参数
    betas,                # 从DiffusionWrapper获取的betas参数
    T=1000,               # 扩散步数
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    独立的条件扩散采样函数
    """
    model.eval()  # 切换到评估模式
    num_samples = labels.shape[0]
    
    with torch.no_grad():
        # 初始化随机噪声
        x_t = torch.randn((num_samples, sample_shape), device=device)
        
        # 反向扩散过程
        for t in reversed(range(T)):
            # 准备时间步张量
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            eps_pred = model(x_t, labels, t_batch)
            
            # 计算alpha相关参数
            alpha_bar_t = alpha_bar[t]
            alpha_bar_prev = alpha_bar[t-1] if t > 0 else 1.0
            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = betas[t]
            
            # 计算均值
            mu = (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t)
            
            # 添加噪声（最后一步不添加）
            if t > 0:
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mu + sigma_t * z
            else:
                x_t = mu
    
    model.train()  # 恢复训练模式
    return x_t.cpu()  # 返回CPU tensor方便后续处理

# 示例用法
if __name__ == "__main__":
    import os
    from torch.utils.data import TensorDataset, DataLoader
    # 超参数
    L = 2    # 潜在空间维度
    num_classes = 2  # 类别数
    batch_size = 32
    num_epochs = 1000
    T = 1000        # 扩散步数
    
    # 初始化模型和扩散包装器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalDiffusionModel(input_dim=L, num_classes=num_classes).to(device)
    diffusion = DiffusionWrapper(model, T=T)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 数据集
    import pandas as pd
    import numpy as np
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, f"../newcircuit/yycluster/latent_{L}_for_diffusion.csv")
    df = pd.read_csv(csv_path)
    # df = pd.read_csv("../latentspace/ising_ground_state_64_3_for_diffusion.csv")
    d = df.to_numpy()
    d = torch.tensor(d, dtype=torch.float32)
    l = torch.tensor([0]*801 + [1]*801, dtype=torch.long)  # 801个0和81个1
    dataset = TensorDataset(d, l)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for x0, labels in dataloader:
            x0 = x0.to(device)
            labels = labels.to(device)
            
            loss = diffusion.train_step(x0, labels, optimizer)
            total_loss += loss
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")


# sample data from the trained model
    # 生成样本
    labels = torch.tensor([0] * 1000 + [1] * 1000).to(device)  # 生成10个每个类别的样本
    sample_shape = L
    samples = conditional_diffusion_sample(
        model,
        labels,
        sample_shape,
        diffusion.alpha_bar,
        diffusion.betas,
        T=T,
        device=device
    )
    
    # 打印生成的样本形状
    print("Generated samples shape:", samples.shape)
    # save the generated samples
    samples = samples.cpu().numpy()
    df = pd.DataFrame(samples)
    save_path = os.path.join(current_dir, f"../newcircuit/yycluster/diffsuion_samples_{L}.csv")
    df.to_csv(save_path, index=False)