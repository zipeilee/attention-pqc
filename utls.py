import os
import torch
import pandas as pd
import re
from torch import Tensor
from torch.utils.data import TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def scale_2pi_to_neg_to_one(x):
    return x / (2 * torch.pi)


def scale_2pi_to_zero_to_one(x):
    return (x + 2 * torch.pi) / (4 * torch.pi)

def inverse_scale_2pi_zero_to_one(x):
    return x * (4 * torch.pi) - 2 * torch.pi

def inverse_scale_2pi_to_neg_to_one(x):
    return x * 2 * torch.pi

def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())

def inverse_min_max_scale(x, min, max):
    return x * (max - min) + min

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

from typing import Tuple

def read_file(dir) -> Tuple[Tensor, Tensor]:

    files = sorted(os.listdir(dir), key=natural_sort_key)

    labels = []
    for file in files:
        base_name = os.path.splitext(file)[0]
        labels.append(base_name)
    
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}

    tensor_dataset = torch.tensor([], dtype=torch.float32)
    labels_dataset = torch.tensor([], dtype=torch.int64)
    for file in files:
        file_path = os.path.join(dir, file)
        df = pd.read_csv(file_path)
        matrix = df.to_numpy()
        tensor_data = torch.tensor(matrix, dtype=torch.float32).T
        tensor_dataset = torch.cat((tensor_dataset, tensor_data), 0)
        base_name = os.path.splitext(file)[0]
        labels_dataset = torch.cat([
            labels_dataset,
            torch.full((tensor_data.shape[0],), 
                      label_to_id[base_name], 
                      dtype=torch.int64)
        ])
    
    
    return tensor_dataset, labels_dataset

def comb_dataset(tensor_data: Tensor, label_set: Tensor) -> TensorDataset:
    return TensorDataset(tensor_data, label_set)


def plot_latent(tensor_set, labels_set, 
                title = 'Latent Space',
                xlabel = 'z1',
                ylabel = 'z2',
                cbars = 'Labels',
                colors = 'plasma',
                grid = True,
                alpha=0.7, s=50, marker='D'):
    num_labels = len(labels_set.unique())
    if torch.is_tensor(tensor_set):
        tensor_set = tensor_set.detach().numpy()
        labels_set = labels_set.detach().numpy()
        
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette(colors, num_labels)
    scatter = sns.scatterplot(
        x = tensor_set[:, 0],
        y = tensor_set[:, 1],
        hue = labels_set,
        palette=palette,
        alpha = alpha,
        legend = False,
        s = s,
        marker=marker

    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    norm = plt.Normalize(labels_set.min(), labels_set.max())
    sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(cbars)
    plt.tick_params(direction="in")
    plt.grid(grid)
    plt.show()

def plot_dim_1(tensor_set, labels_set, 
               xlabel = 'J\'/J',
               ylabel = 'z',
               title='one dim kpca',cmap='plasma',marker='o',s=15):
    colors = np.linspace(0, 1, len(labels_set.unique()))
    plt.figure(figsize=(10, 8))
    plt.scatter(labels_set, tensor_set, c=colors, cmap=cmap, marker=marker, s=s)
    plt.xlim(labels_set.min(), labels_set.max()+0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=xlabel)
    plt.show()





def plot_loss(loss_history,epochs, xlog_scale=True):
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', markersize=1.5, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss Curve')
    plt.legend()
    if xlog_scale:
        plt.xscale('log')
    plt.grid(True)
    plt.show()




def sliding_tensor(data: Tensor, window_size:int, stride=1):
    N, L = data.shape
    half_window = window_size // 2

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    if N < window_size:
        raise ValueError('window_size must be less than N')
    
    center_indices = torch.arange(half_window, N - half_window, stride)

    high_order_tensor = torch.stack([
        data[i-half_window:i+half_window+1] 
        for i in center_indices
    ], dim=0)

    return high_order_tensor, center_indices


def plot_means_with_vars(means, vars, l, figsize=(10, 8),
                            xlabel = 'J\'/J',
                            ylabel = 'means',
                            title='one dim kpca',
                            cmap='plasma',
                            marker='^',
                            alpha=0.3,
                            ifgrid=True,
                            scatteralpha=1,
                            s=50):
    means = means.flatten().detach().numpy()
    vars = vars.flatten().detach().numpy()
 
    colors = np.linspace(0, 1, len(l.unique()))
    plt.figure(figsize=figsize)

    plt.plot(l, means,  color='gray',alpha=alpha)

    plt.scatter(l, means, c=colors, cmap=cmap, marker=marker, s=s, alpha=scatteralpha)
    
    plt.fill_between(l,  -vars,    vars, color='gray',
                      alpha=alpha,
                     )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label=xlabel)
    plt.grid(ifgrid)
    plt.tick_params(axis='both', direction='in')
    plt.show()