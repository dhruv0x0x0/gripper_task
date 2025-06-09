import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from typing import Tuple, Sequence, Dict, Union, Optional, Callable


# -----------------------------------------------------------------------------
# Diffusion-based language generation training script
# Non-autoregressive: fixed-length sequences, noisy (masked) inputs,
# model learns to denoise. Also includes an inference sampler.
# -----------------------------------------------------------------------------

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
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
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# Inference sampler: reverse diffusion via Gumbel-max & confidence-based unmasking
# -----------------------------------------------------------------------------

def add_gumbel_noise(logits: torch.Tensor, temperature: float):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    rem = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :rem[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def generate(model, prompt: torch.Tensor, cond: torch.Tensor, 
             steps: int = 128, gen_length: int = 10,
             temperature: float = 0.1,
             cfg_scale: float = 4.0,
             remasking: str = 'low_confidence',
             mask_id: int = None, device = 'cpu'):
    device = device
    mask_id = mask_id #or model.config.mask_token_id

    x = torch.full((1, gen_length), mask_id, dtype=torch.long, device=device)
    ss = 0
    if prompt:
       x[0, :prompt.size(1)] = prompt
       ss = prompt.size(1)
    prompt_mask = x != mask_id
    mask_idx = x == mask_id
    num_tokens = get_num_transfer_tokens(mask_idx, steps)
    for t in range(steps):
        if cfg_scale > 0:
            uncond = x.clone()
            uncond[prompt_mask] = mask_id
            x_ = torch.cat([x, uncond], dim=0)
            cond_cat = torch.cat([cond, cond], dim=0)
            logits = model(x_, cond_cat)
            logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale+1)*(logits - uncond_logits)
        else:
            logits = model(x,cond)

        logits_noise = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_noise,dim=-1)

        if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        else:
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        
       # x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

        x0 = torch.where(mask_idx, x0, x)
        confidence = torch.where(mask_idx, x0_p, -np.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_tokens[j, t])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    return x

