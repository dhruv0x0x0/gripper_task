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

import os
import json
import numpy as np
import torch
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

class TemporalBPEProcessor:
    def __init__(
        self,
        scale: float = 100.0,
        min_token: int = 0,
        time_horizon: int | None = None,
        state_dim: int | None = None,
        normalization: str = "zscore",  # 'zscore' or 'minmax'
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        min_val: np.ndarray | None = None,
        max_val: np.ndarray | None = None,
    ):
        self.scale = scale
        self.min_token = min_token
        self.time_horizon = time_horizon
        self.state_dim = state_dim
        self.normalization = normalization
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.called_time_horizon = time_horizon
        self.called_state_dim = state_dim
        self.T = 16
        self.mask_token_id = 0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            return (x - self.mean) / self.std
        elif self.normalization == "minmax":
            mnmx = self.max_val - self.min_val
            mnmx[mnmx<=1e-2]= 1
            return (x - self.min_val) / mnmx
        return x

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            return x * self.std + self.mean
        elif self.normalization == "minmax":
            mnmx = self.max_val - self.min_val
            mnmx[mnmx<=1e-2]= 1
            return x * (mnmx) + self.min_val
        return x

    def __call__(
        self,
        state_seq: np.ndarray,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ):
        """
        Tokenize a batch of state sequences via DCT quantization.

        Args:
            state_seq: np.ndarray of shape [B, T, D] or [T, D]
            padding: pad option (currently not used)
            truncation: truncate option (currently not used)
            max_length: length to pad or truncate to if return_tensors='pt'
            return_tensors: 'pt' for PyTorch tensors
        """
        if state_seq.ndim == 2:
            state_seq = state_seq[None, ...]
        #print(state_seq.shape)
        batch_size, T, D = state_seq.shape
        self.called_time_horizon = T
        self.called_state_dim = D

        norm_seq = self.normalize(state_seq)
        coeff = dct(norm_seq, axis=1, norm='ortho')
        q = np.around(coeff * self.scale).astype(int)
        # for i in q[0]:
        #     print(i)

        tokens: list[list[int]] = []
        for b in range(batch_size):
            flat = (q[b].flatten() - self.min_token).clip(min=0).clip(max=200)
            tokens.append(flat.tolist())

        return torch.tensor(tokens)+1

    def decode(
        self,
        token_ids: list[list[int]],
        state_dim: int | None = None,
    ) -> np.ndarray:
        D = state_dim or self.called_state_dim
        token_ids = (token_ids-1).clip(min = 0)
        decoded_seqs = []
        #print('min', self.min_token)
        for ids in token_ids:
            arr = np.array(ids) + self.min_token
            arr = arr.reshape(-1, D)
            coeff = arr.astype(float) / self.scale
            rec = idct(coeff, axis=0, norm='ortho')
            rec = self._denormalize(rec)
            decoded_seqs.append(rec)
        return np.stack(decoded_seqs)

    @classmethod
    def fit_from_npz(
        cls,
        data_dir: str,
        num_episodes: int = -1,
        scale: float = 10.0,
        normalization: str = "zscore",
        T = 32
    ) -> "TemporalBPEProcessor":
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        episodes = meta['episodes']
        if num_episodes == -1 or num_episodes > len(episodes):
            num_episodes = len(episodes)

        pose_seqs = []
        grip_seqs = []
        for ep in episodes:
            arr = np.load(os.path.join(data_dir, ep['file']))
            pose_seq = arr['pose']
            grip_seq = arr['grip'][..., None]
            # if len(seqs)==0:
            #     seqs=np.concatenate([pose_seq, grip_seq], axis=1)
            #x = np.concatenate([pose_seq, grip_seq], axis=1)
            grip_seqs.append(grip_seq)
            pose_seqs.append(pose_seq)#np.concatenate([seqs, x], axis=0)
        
        pose_seqs = np.concatenate(pose_seqs, axis=0)
        grip_seqs = np.concatenate(grip_seqs, axis=0)
        pose_seqs = mat_to_pose9d(pose_seqs.reshape(-1,4,4))
        all_states = np.concatenate([pose_seqs, grip_seqs], axis= -1)
        print(len(all_states))
        mean = all_states.mean(axis=0)
        std = all_states.std(axis=0)
        std[std <= 1e-5] = 1.0
        min_val = all_states.min(axis=0)
        max_val = all_states.max(axis=0)
        #print(all_states.shape)
        if normalization == "zscore":
            normed_seqs = [(s - mean) / std for s in all_states]
        else:
            normed_seqs = [(s - min_val) / (max_val - min_val + 1e-8) for s in all_states]

        # # Determine token range without BPE
        all_coeffs = [dct(s, axis=0, norm='ortho').flatten() for s in normed_seqs]
        scaled_vals = np.around(np.concatenate(all_coeffs) * scale)
        min_token = int(scaled_vals.min())

        D = all_states[0].shape
        print(scale,
            min_token,
            T,
            D,
           normalization,
            mean,
           std,
            min_val,
            max_val,)
        print('done')
        return cls(
            scale=scale,
            min_token=min_token,
            time_horizon= T,
            state_dim=D,
            normalization=normalization,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
        )

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "normalization_stats.npz"),
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            scale=self.scale,
            min_token=self.min_token,
            time_horizon=self.called_time_horizon,
            state_dim=self.called_state_dim,
            normalization=self.normalization,
        )

    @classmethod
    def load(cls, save_dir: str) -> "TemporalBPEProcessor":
        stats = np.load(os.path.join(save_dir, "normalization_stats.npz"))
        return cls(
            scale=float(stats["scale"]),
            min_token=int(stats["min_token"]),
            time_horizon=int(stats["time_horizon"]),
            state_dim=int(stats["state_dim"]),
            normalization=str(stats["normalization"]),
            mean=stats["mean"],
            std=stats["std"],
            min_val=stats["min_val"],
            max_val=stats["max_val"],
        )

    def plot_coeff_histogram(self, data_dir: str, episode_index: int, bins: int = 50):
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        ep = meta['episodes'][episode_index]
        arr = np.load(os.path.join(data_dir, ep['file']))
        pose_seq = arr['pose']
        grip_seq = arr['grip'][..., None]
        seq = np.concatenate([pose_seq, grip_seq], axis=1)

        norm_seq = self.normalize(seq)
        coeffs = dct(norm_seq, axis=0, norm='ortho').flatten()
        plt.figure(figsize=(8, 4))
        plt.hist(coeffs, bins=bins)
        plt.title(f"Normalized DCT Coeffs Histogram — Episode {episode_index}")
        plt.xlabel("Normalized Coefficient")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

def train_tokeniser(data_dir: str = 'out_dataset_bottle', T=16):
    processor = TemporalBPEProcessor.fit_from_npz(
        data_dir, num_episodes=-1, scale=10, normalization="zscore", T=T
    )
    processor.save("saved_processor")