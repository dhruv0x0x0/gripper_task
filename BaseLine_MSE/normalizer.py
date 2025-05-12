import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Union
from utils import mat_to_pose9d

def _fit(data: Union[torch.Tensor, np.ndarray],
         last_n_dims=1,
         dtype=torch.float32,
         mode='limits',
         output_max=1.0,
         output_min=-1.0,
         range_eps=1e-4,
         fit_offset=True) -> nn.ParameterDict:
    """
    Compute and store normalization parameters for a tensor or array.
    Supports 'limits' (min-max) or 'gaussian' (z-score) normalization.
    """
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # Flatten last dimensions to one
    dim = 1
    if last_n_dims > 0:
        dim = int(np.prod(data.shape[-last_n_dims:]))
    data = data.reshape(-1, dim)

    # Compute stats
    input_min, _ = data.min(dim=0)
    input_max, _ = data.max(dim=0)
    input_mean = data.mean(dim=0)
    input_std = data.std(dim=0)

    # Compute scale and offset
    if mode == 'limits':
        if fit_offset:
            input_range = input_max - input_min
            ignore = input_range < range_eps
            input_range[ignore] = (output_max - output_min)
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            # constant dims -> midpoint
            offset[ignore] = ((output_max + output_min) / 2) - input_min[ignore]
        else:
            # zero-centered
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.max(input_max.abs(), input_min.abs())
            ignore = input_abs < range_eps
            input_abs[ignore] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:  # gaussian
        ignore = input_std < range_eps
        std = input_std.clone()
        std[ignore] = 1.0
        scale = 1.0 / std
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    params = nn.ParameterDict({
        'scale': nn.Parameter(scale),
        'offset': nn.Parameter(offset),
        'input_stats': nn.ParameterDict({
            'min': nn.Parameter(input_min),
            'max': nn.Parameter(input_max),
            'mean': nn.Parameter(input_mean),
            'std': nn.Parameter(input_std)
        })
    })
    for p in params.parameters():
        p.requires_grad_(False)
    return params


def _normalize(x: Union[torch.Tensor, np.ndarray], params: nn.ParameterDict, forward=True) -> torch.Tensor:
    """
    Apply or invert normalization using stored params.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    return x.reshape(shape)


class DataNormalizer:
    """
    Full pipeline: fit from data_dir, save/load stats, normalize and unnormalize.
    """
    def __init__(self,
                 mode: str = 'limits',
                 output_min: float = -1.0,
                 output_max: float = 1.0,
                 range_eps: float = 1e-4,
                 fit_offset: bool = True,
                 last_n_dims: int = 1,
                 dtype=torch.float32,
                 normalization: str = 'limits'):
        self.mode = mode
        self.output_min = output_min
        self.output_max = output_max
        self.range_eps = range_eps
        self.fit_offset = fit_offset
        self.last_n_dims = last_n_dims
        self.dtype = dtype
        self.params = None
        self.normalization = normalization

    @classmethod
    def fit_from_npz(cls,
                     data_dir: str,
                     sample_limit: int = -1) -> 'DataNormalizer':
        """
        Reads meta.json, loads arrays, concatenates pose and grip sequences,
        computes stats, and fits normalization parameters.
        """
        # Load metadata
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        episodes = meta['episodes']
        if sample_limit > 0:
            episodes = episodes[:sample_limit]

        # Collect data
        pose_list = []
        grip_list = []
        for ep in episodes:
            arr = np.load(os.path.join(data_dir, ep['file']))
            pose_list.append(arr['pose'])
            grip_list.append(arr['grip'][..., None])

        pose_seqs = np.concatenate(pose_list, axis=0)
        grip_seqs = np.concatenate(grip_list, axis=0)
        pose_seqs = mat_to_pose9d(pose_seqs.reshape(-1,4,4))
        all_states = np.concatenate([pose_seqs, grip_seqs], axis=-1)

        # Fit
        normalizer = cls(mode='limits', last_n_dims=all_states.ndim - 1)
        normalizer.params = _fit(all_states,
                                 last_n_dims=all_states.ndim - 1,
                                 mode=normalizer.mode,
                                 output_max=normalizer.output_max,
                                 output_min=normalizer.output_min,
                                 range_eps=normalizer.range_eps,
                                 fit_offset=normalizer.fit_offset,
                                 dtype=normalizer.dtype)
        return normalizer

    def save(self, save_dir: str):
        """
        Save normalization stats to NPZ file.
        """
        os.makedirs(save_dir, exist_ok=True)
        stats = {k: v.detach().cpu().numpy() for k, v in self.params['input_stats'].items()}
        np.savez(os.path.join(save_dir, 'normalization_stats.npz'),
                 mode=self.mode,
                 output_min=self.output_min,
                 output_max=self.output_max,
                 range_eps=self.range_eps,
                 fit_offset=int(self.fit_offset),
                 last_n_dims=self.last_n_dims,
                 scale=self.params['scale'].detach().cpu().numpy(),
                 offset=self.params['offset'].detach().cpu().numpy(),
                 **stats)

    @classmethod
    def load(cls, load_dir: str) -> 'DataNormalizer':
        """
        Load stats from NPZ and reconstruct normalizer.
        """
        path = os.path.join(load_dir, 'normalization_stats.npz')
        data = np.load(path)
        normalizer = cls(mode=data['mode'],
                         output_min=float(data['output_min']),
                         output_max=float(data['output_max']),
                         range_eps=float(data['range_eps']),
                         fit_offset=bool(data['fit_offset']),
                         last_n_dims=int(data['last_n_dims']),
                         dtype=torch.float32)
        # Restore params
        pd = nn.ParameterDict({
            'scale': nn.Parameter(torch.from_numpy(data['scale'])),
            'offset': nn.Parameter(torch.from_numpy(data['offset'])),
            'input_stats': nn.ParameterDict({
                'min': nn.Parameter(torch.from_numpy(data['min'])),
                'max': nn.Parameter(torch.from_numpy(data['max'])),
                'mean': nn.Parameter(torch.from_numpy(data['mean'])),
                'std': nn.Parameter(torch.from_numpy(data['std'])),
            })
        })
        for p in pd.parameters(): p.requires_grad_(False)
        normalizer.params = pd
        return normalizer

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Normalize input data.
        """
        if self.params is None:
            raise RuntimeError("Normalizer has not been fitted or loaded.")
        return _normalize(x, self.params, forward=True)

    def unnormalize(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Reverse normalization.
        """
        if self.params is None:
            raise RuntimeError("Normalizer has not been fitted or loaded.")
        return _normalize(x, self.params, forward=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train, save, load, and apply a data normalizer.")
    parser.add_argument('data_dir', help='Directory containing meta.json and .npz files')
    parser.add_argument('--save_dir', default='./norm_stats', help='Directory to save normalization stats')
    parser.add_argument('--mode', choices=['limits', 'gaussian'], default='limits')
    parser.add_argument('--sample_limit', type=int, default=-1, help='Limit number of episodes')
    args = parser.parse_args()

    # Fit and save
    normalizer = DataNormalizer.fit_from_npz(args.data_dir, sample_limit=args.sample_limit)
    print("Fitted normalizer with mode=", args.mode)
    normalizer.save(args.save_dir)
    print(f"Saved stats to {args.save_dir}")

    # Example load and usage
    loaded = DataNormalizer.load(args.save_dir)
    print("Loaded normalizer. Sample normalization:")
    # Suppose we take first episode data for testing
    sample = np.load(os.path.join(args.data_dir, json.load(open(os.path.join(args.data_dir, 'meta.json')))['episodes'][0]['file']))
    test_state = np.concatenate([sample['pose'][0:1], sample['grip'][0:1][..., None]], axis=-1)
    normed = loaded.normalize(test_state)
    unnormed = loaded.unnormalize(normed)
    print("Original:", test_state[0])
    print("Normalized:", normed[0])
    print("Reconstructed:", unnormed[0])
