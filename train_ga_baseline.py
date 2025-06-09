import torch
import torchvision
#import r3m
work = ''
from tokeniser import train_tokeniser
train_tokeniser()
import os
import json
import numpy as np
import torch
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

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
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimpleConfig:
    """Simple configuration class with essential parameters"""
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 10
    n_layers: int = 12
    mlp_ratio: float = 4.0
    max_seq_length: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0
    
    @property
    def head_dim(self):
        return self.d_model // self.n_heads
    
    @property
    def hidden_size(self):
        return int(self.mlp_ratio * self.d_model)

class RMSLayerNorm(nn.Module):
    """RMS layer normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings"""
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self._compute_cos_sin_cache(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def _compute_cos_sin_cache(self, seq_len):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        seq = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, seq_len, self.dim)
        sin = emb.sin().view(1, 1, seq_len, self.dim)
        return cos, sin
        
    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.size(2)
        if seq_len > self.max_seq_len:
            cos, sin = self._compute_cos_sin_cache(seq_len)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q_rot = torch.cat([-q2, q1], dim=-1)
        k_rot = torch.cat([-k2, k1], dim=-1)
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        return q, k

class AttentionBlock(nn.Module):
    """Self-attention block similar to LLaMA"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.norm = RMSLayerNorm(config.d_model)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            config.head_dim, 
            max_seq_len=config.max_seq_length,
            theta=config.rope_theta
        )
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        bsz, seqlen, _ = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(bsz, seqlen, self.config.n_heads, self.config.head_dim).transpose(1,2)
        k = self.k_proj(h).view(bsz, seqlen, self.config.n_heads, self.config.head_dim).transpose(1,2)
        v = self.v_proj(h).view(bsz, seqlen, self.config.n_heads, self.config.head_dim).transpose(1,2)
        q, k = self.rope(q, k, seqlen)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.config.head_dim)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -float('inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(bsz, seqlen, self.config.d_model)
        out = self.o_proj(out)
        return x + self.dropout(out)

class FeedForwardBlock(nn.Module):
    """Feed-forward block with SwiGLU activation"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.norm = RMSLayerNorm(config.d_model)
        self.w1 = nn.Linear(config.d_model, config.hidden_size, bias=False)
        self.w2 = nn.Linear(config.d_model, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        h = self.norm(x)
        h1 = self.w1(h)
        h2 = self.w2(h)
        hidden = F.silu(h1) * h2
        out = self.w3(hidden)
        return x + self.dropout(out)

class PerceiverResampler(nn.Module):
    """Resample a [B, D] cond vector into [B, num_latents, d_model] via cross-attention"""
    def __init__(self, cond_dim=522, d_model=768, num_latents=64, n_heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model))
        self.proj = nn.Linear(cond_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, cond):
        # cond: [B, cond_dim]
        B = cond.shape[0]
        # project cond to d_model and add seq dim
        kv = self.proj(cond).unsqueeze(1)           # [B, 1, d_model]
        lat = self.latents.expand(B, -1, -1)        # [B, num_latents, d_model]
        # cross-attn: lat queries, cond keys/values
        out, _ = self.cross_attn(lat, kv, kv)       # [B, num_latents, d_model]
        return out

class TransformerBlock(nn.Module):
    """Combined self-attn + cross-attn + FFN"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.attn = AttentionBlock(config)
        self.ffn  = FeedForwardBlock(config)
        self.cross_dropout = nn.Dropout(config.dropout)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)

    def forward(self, x, cond_seq):
        # x: [B, seq_len, d_model]
        # cond_seq: [B, num_latents, d_model]
        x = self.attn(x)
        attn_out, _ = self.cross_attn(x, cond_seq, cond_seq)
        x = x + self.cross_dropout(attn_out)
        x = self.ffn(x)
        return x

class SimpleLLaDAModel(nn.Module):
    """LLaDA with Perceiver resampler + cross-attn"""
    def __init__(self, config: SimpleConfig, cond_dim=522, num_latents=64):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.resampler = PerceiverResampler(cond_dim, config.d_model, num_latents, config.n_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSLayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, input_ids, cond):
        # input_ids: [B, seq_len]
        # cond:      [B, cond_dim]
        x = self.token_emb(input_ids)
        cond_seq = self.resampler(cond)
        for blk in self.blocks:
            x = blk(x, cond_seq)
        x = self.norm(x)
        return self.head(x)
def create_model(vocab_size=32000, d_model=768, n_heads=12, n_layers=12,
                 cond_dim=542, num_latents=64):
    cfg = SimpleConfig(vocab_size=vocab_size, d_model=d_model,
                       n_heads=n_heads, n_layers=n_layers)
    return SimpleLLaDAModel(cfg, cond_dim=cond_dim, num_latents=num_latents)

import os
import json
import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch

np.set_printoptions(precision=3, suppress=True)

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
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

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
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
from torch.utils.data import Dataset

class ManiSkillSequenceDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, state_horizon: int = 16, past_poses: int = 1):
        self.data_dir = data_dir
        self.transform = transform
        self.state_horizon = state_horizon
        self.past_poses = past_poses
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.episodes = meta['episodes']
        lengths = [ep['length'] for ep in self.episodes]
        self.cumlen = np.cumsum([0] + lengths)

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx: int):
        ep = int(np.searchsorted(self.cumlen, idx, side='right') - 1)
        step = idx - self.cumlen[ep]
        arr = np.load(os.path.join(self.data_dir, self.episodes[ep]['file']))
        imgs = arr['img']
        img_hist = []
        for i in range(self.past_poses, -1, -1):
            past_idx = max(step - i, 0)
            img = imgs[past_idx]
            if self.transform:
                img_t = self.transform(img)
            else:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.)
            img_hist.append(img_t)
        img_hist = torch.stack(img_hist, dim=0)
        all_poses = mat_to_pose10d(arr['pose'].reshape(-1, 4, 4))
        all_grips = arr['grip']
        pose_hist = []
        for i in range(self.past_poses, -1, -1):
            past_idx = max(step - i, 0)
            pose_hist.append(all_poses[past_idx])
        past_poses_arr = np.stack(pose_hist, axis=0).astype(np.float32)
        grip_hist = []
        for i in range(self.past_poses, -1, -1):
            past_idx = max(step - i, 0)
            grip_hist.append([all_grips[past_idx]])
        past_grips_arr = np.stack(grip_hist, axis=0).astype(np.float32)
        horizon = []
        feat_dim = all_poses.shape[1]
        zero_feat = np.zeros((feat_dim,), dtype=np.float32)
        zero_grip = np.zeros((1,), dtype=np.float32)
        for i in range(self.state_horizon):
            future_idx = step + 1 + i
            if future_idx < len(all_poses):
                p = all_poses[future_idx]
                g = np.array([all_grips[future_idx]], dtype=np.float32)
            else:
                p, g = zero_feat, zero_grip
            horizon.append(np.concatenate([p, g], axis=0))
        target_state = np.stack(horizon, axis=0)
        return {
            'obs': {
                'image': img_hist,
                'pose': torch.from_numpy(past_poses_arr),
                'gripper': torch.from_numpy(past_grips_arr)
            },
            'target_state': target_state
        }


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
# from model import create_model
# from data import ManiSkillSequenceDataset
# from tokeniser import TemporalBPEProcessor
# from vision_model_getter import get_resnet
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

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

import copy
import torch
import matplotlib.pyplot as plt
from torch.nn.modules.batchnorm import _BatchNorm

class EMAModel:
    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2/3,
        min_value=0.0,
        max_value=0.9999
    ):
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.to(ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(ema_param.dtype), alpha=1 - self.decay)
        self.optimization_step += 1
        
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

class DiffusionTrainer2:
    def __init__(self,
                 tokeniser,
                 seq_len: int = 128,
                 schedule_steps: int = 1000,
                 vocab_size=1000,
                 d_model=768,
                 n_heads=12,
                 n_layers=12,
                 device: str = 'cuda',
                 load_path: str = None):  # <-- Added load_path
        self.tokenizer = tokeniser
        self.seq_len = seq_len
        self.T = schedule_steps
        self.device = device
        self.vocab_size = vocab_size
        self.mask_id = 1             # the token ID you want to mask out

        # Define your adaptive softmax here once, so it shares parameters across calls
        self.cutoffs = [100, 250, 400]    # example cutoffs for a 500-word vocab
        
       
        resnet = replace_bn_with_gn(get_resnet('resnet18'))
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        self.model = nn.ModuleDict({
            'vision_encoder': resnet,
            'lldm': create_model(vocab_size=500, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                 cond_dim=2*512*49+2*10, num_latents=64).to(device)
        })

        if load_path and os.path.isfile(load_path):  # <-- Load model if specified
            print(f"Loading model from {load_path}")
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))

        self.mask_id = self.tokenizer.mask_token_id
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        
    def linear_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() + 1) / self.T
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor):
        p_mask = (t.float() + 1) / self.T
        rand = torch.rand_like(x_start.float(), device=self.device)
        mask_indices = rand < p_mask
        x_noisy = x_start.clone()
        x_noisy[mask_indices] = self.mask_id
        return x_noisy, mask_indices

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target.view(-1)
        losses = loss_fct(logits_flat, target_flat).view_as(target)
        masked = mask_indices.float()
        return (losses * masked).sum() / masked.sum().clamp(min=1)
    def compute_loss_adaptive(self,
                              hidden_states: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch_size, seq_len, hidden_dim)
        target:        (batch_size, seq_len), containing token IDs in [0, vocab_size)
        """
        # flatten batch & sequence dims
       
        batch_size, seq_len, h= hidden_states.size()
        hidden_flat = hidden_states.view(-1, h)    # (B·S, H)
        target_flat = target.view(-1)                            # (B·S,)
        adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=h,
            n_classes=self.vocab_size,
            cutoffs=self.cutoffs,
            div_value=4.0,
            head_bias=True
        )
        # compute a boolean mask for positions where target == mask_id
        mask_flat = (target_flat == self.mask_id)                # (B·S,)

        # select only masked positions
        masked_hidden = hidden_flat[mask_flat]
        masked_target = target_flat[mask_flat]

        # if there are no masked tokens, return zero loss
        if masked_target.numel() == 0:
            return torch.tensor(0., device=hidden_states.device)

        # compute the adaptive softmax loss
        out, loss = adaptive_softmax(masked_hidden, masked_target)
        return loss
    def train(self, dataloader, batch_size: int = 32, epochs: int = 10, save_path: str = 'model'): 
        self.model.to(self.device)
        self.model.train()
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * epochs
        )

        ema = EMAModel(self.model, update_after_step=0, inv_gamma=1.0, power=2/3)

        epoch_losses = []
        print('training started')

        for epoch in range(epochs):
            total_loss = 0.0
            for i, batch in enumerate(dataloader):
                # Prepare inputs
                rgb_img = torch.tensor(batch['obs']['image']).to(self.device)
                gripper_state = torch.cat((batch['obs']['pose'], batch['obs']['gripper']), dim=-1)
                qq,yy,ss = gripper_state.shape
                gripper_state = gripper_state.reshape(-1, yy*ss)
                gripper_state = torch.tensor(gripper_state).to(self.device)
                target_states = batch['target_state']
                rgb_img = torch.tensor(rgb_img).view(-1, 3, 224, 224)
                image_features = self.model['vision_encoder'](rgb_img)
                B, C, H, W = image_features.shape
                image_features = image_features.reshape(B, C*H*W)#.permute(0, 2, 1) 
                image_features = image_features.reshape(-1, 2*C*H*W)
                #print(ima)
                #image_features = image_features.unsqueeze(0)
                #print("img", image_features.shape)
                obs_features = torch.cat([image_features, gripper_state], dim=-1)
                inputs = self.tokenizer(target_states).to(self.device)

                t = torch.randint(0, self.T, inputs.shape, device=self.device)
                x_noisy, mask_indices = self.q_sample(inputs, t)
                #print(x_noisy.shape, obs_features.shape)
                logits = self.model['lldm'](x_noisy, obs_features)
                loss = self.compute_loss(logits, inputs, mask_indices)

                if i % 10 == 0:
                    print(f"Batch {i} loss: {loss.item():.4f}")

                # Backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lr_scheduler.step()

                # Update EMA
                ema.step(self.model)

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Save both original and EMA models
            torch.save(self.model.state_dict(), f"{save_path}img_epoch{epoch+1}.pt")
            torch.save(ema.averaged_model.state_dict(), f"{save_path}img_ema_epoch{epoch+1}.pt")

        # Plot training loss
        plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.grid()
        plt.savefig("training_loss.png")
        plt.show()
from torchvision import transforms

if  __name__== '__main__':
    mode = 'train'
    batch_size = 64
    epochs = 50
    tokeniser_path = "saved_processor" 
    print('loading_dataset')
    ds = ManiSkillSequenceDataset('out_dataset_bottle', transform=transforms.ToTensor())
    print('loaded')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    tokeniser = TemporalBPEProcessor.load(tokeniser_path)
    trainer = DiffusionTrainer2(tokeniser=tokeniser, vocab_size=500, device= 'cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(loader, batch_size=batch_size, epochs=epochs)

