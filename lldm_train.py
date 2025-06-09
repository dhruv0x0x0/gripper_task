import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

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
    schedule = base.repeat(1, steps)
    for i in range(mask_num.size(0)):
        schedule[i, :rem[i]] += 1
    return schedule.long()

@torch.no_grad()
def generate(model, prompt: torch.Tensor,
             steps: int = 128, gen_length: int = 128,
             block_length: int = 32,
             temperature: float = 0.0,
             cfg_scale: float = 0.0,
             remasking: str = 'low_confidence',
             mask_id: int = None):
    device = model.device
    mask_id = mask_id or model.config.mask_token_id
    B = gen_length // block_length
    steps_per_block = steps // B

    # init: keep prompt, mask the generation region
    x = torch.full((1, prompt.size(1)+gen_length), mask_id, dtype=torch.long, device=device)
    x[0, :prompt.size(1)] = prompt
    prompt_mask = x != mask_id

    for b in range(B):
        start = prompt.size(1) + b*block_length
        end = start + block_length
        mask_idx = x[:, start:end] == mask_id
        num_tokens = get_num_transfer_tokens(mask_idx, steps_per_block)
        for t in range(steps_per_block):
            mask_all = x == mask_id
            # classifier-free guidance
            if cfg_scale > 0:
                uncond = x.clone()
                uncond[prompt_mask] = mask_id
                cat = torch.cat([x, uncond], dim=0)
                logits = model(cat).logits
                cond, uncond_logits = logits.chunk(2, dim=0)
                logits = uncond_logits + (cfg_scale+1)*(cond - uncond_logits)
            else:
                logits = model(x).logits

            logits_noise = add_gumbel_noise(logits, temperature)
            x0 = logits_noise.argmax(dim=-1)

            if remasking == 'low_confidence':
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                confid = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                confid = torch.rand_like(x, dtype=torch.float64)
            confid[:, :start+block_length] = -np.inf

            # pick top-k_t
            k_t = num_tokens[:, t]
            transfer = torch.zeros_like(x, dtype=torch.bool)
            for i in range(x.size(0)):
                _, idx = torch.topk(confid[i], k=k_t[i].item())
                transfer[i, idx] = True

            x[transfer] = x0[transfer]

    return x

# -----------------------------------------------------------------------------
# Main entry points
#