import torch
from collections import deque

class StateQueue:
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self._buffer = deque(maxlen=k)

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 1:
            raise ValueError(f"Expected 1-D tensor, got shape {tuple(tensor.shape)}")
        self._buffer.append(tensor)

    def get(self) -> torch.Tensor:
        if not self._buffer:
            raise IndexError("Cannot get from an empty StateQueue")

        current = list(self._buffer)
        n = len(current)

        if n < self.k:
            repeats = self.k - n + 1
            oldest = current[0]
            seq = [oldest] * repeats + current[1:]
        else:
            seq = current[-self.k:]

        return torch.cat(seq)

    def __len__(self) -> int:
        return len(self._buffer)
from collections import deque
import torch

class StateQueue_dict:
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self._buffer = deque(maxlen=k)
        self._keys = None

    def update(self, obs: dict) -> None:
      
        if not isinstance(obs, dict):
            raise ValueError(f"Expected obs to be a dict, got {type(obs)}")

        if self._keys is None:
            self._keys = set(obs.keys())
        elif set(obs.keys()) != self._keys:
            raise ValueError(f"Obs keys {set(obs.keys())} do not match expected {self._keys}")

        self._buffer.append(obs.copy())

    def get(self) -> dict:
  
        if not self._buffer:
            raise IndexError("Cannot get from an empty StateQueue")
        current = list(self._buffer)
        n = len(current)
        if n < self.k:
            repeats = self.k - n + 1
            oldest = current[0]
            seq_dicts = [oldest] * repeats + current[1:]
        else:
            seq_dicts = current[-self.k:]
        sample_image = seq_dicts[0].get('image')
        device = sample_image.device if torch.is_tensor(sample_image) else torch.device('cpu')
        out = {}
        for key in self._keys:
            values = [d[key] for d in seq_dicts]
            tensors = []
            for v in values:
                if not torch.is_tensor(v):
                    t = torch.tensor(v, device=device)
                else:
                    t = v.to(device)
                tensors.append(t)
            stacked = torch.stack(tensors, dim=0)
            
            out[key] = stacked.unsqueeze(0)

        return out

    def __len__(self) -> int:
        return len(self._buffer)

import torch
import torchvision

import numpy as np
import torch
from typing import Callable
np.set_printoptions(precision=3, suppress=True)

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

np.set_printoptions(precision=3, suppress=True)
def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose9d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose9d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out