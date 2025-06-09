# import os
# import h5py
# import numpy as np
# from scipy.fft import dct, idct
# # from tokenizers import ByteLevelBPETokenizer
# # from tokenizers.trainers import BpeTrainer
# from mani_skill.utils import io_utils
# from mani_skill.trajectory import utils as trajectory_utils
# import matplotlib.pyplot as plt
# np.set_printoptions(precision=3, suppress=True)
# import torch

# class TemporalBPEProcessor:
#     def __init__(
#         self,
#         bpe_tokenizer: ByteLevelBPETokenizer,
#         scale: float = 100.0,
#         vocab_size: int = 1000,
#         min_token: int = 0,
#         time_horizon: int | None = None,
#         state_dim: int | None = None,
#         normalization: str = "zscore",  # 'zscore' or 'minmax'
#         mean: np.ndarray | None = None,
#         std: np.ndarray | None = None,
#         min_val: np.ndarray | None = None,
#         max_val: np.ndarray | None = None,
#     ):
#         self.bpe = bpe_tokenizer
#         self.scale = scale
#         self.vocab_size = vocab_size
#         self.min_token = min_token
#         self.time_horizon = time_horizon
#         self.state_dim = state_dim
#         self.normalization = normalization
#         self.mean = mean
#         self.std = std
#         self.min_val = min_val
#         self.max_val = max_val
#         self.called_time_horizon = time_horizon
#         self.called_state_dim = state_dim
#         # define pad token id (use min_token if you haven't set a special pad token)
#         self.mask_token_id = 0
        
#         #self.
#     def normalize(self, x):
#         if self.normalization == "zscore":
#             return (x - self.mean) / (self.std)
#         elif self.normalization == "minmax":
#             return (x - self.min_val) / (self.max_val - self.min_val + 1e-8)
#         #print("yo", x.shape)
#         return x

#     def _denormalize(self, x):
#         if self.normalization == "zscore":
#             return x * self.std + self.mean
#         elif self.normalization == "minmax":
#             return x * (self.max_val - self.min_val + 1e-8) + self.min_val
#         return x

#     def __call__(
#         self,
#         state_seq: np.ndarray,
#         padding: bool = False,
#         truncation: bool = False,
#         max_length: int | None = None,
#         return_tensors: str | None = None,
#     ):
#         """
#         Tokenize a batch of state sequences.

#         Args:
#             state_seq: np.ndarray of shape [B, T, D] or [T, D]
#             padding: 'max_length' to pad to max_length, True/False or 'longest' for dynamic padding
#             truncation: True to truncate sequences longer than max_length
#             max_length: length to pad or truncate to
#             return_tensors: 'pt' to return PyTorch tensors

#         Returns:
#             If return_tensors='pt', returns a dict with 'input_ids' and 'attention_mask'.
#             Otherwise, returns (tokens: List[List[int]], time_horizon: int).
#         """
#         # ensure batch dim
#         if state_seq.ndim == 2:
#             state_seq = state_seq[None, ...]

#         batch_size, T, D = state_seq.shape
#         self.called_time_horizon = T
#         self.called_state_dim = D

#         # Normalize and compute DCT per sequence
#         norm_seq = self.normalize(state_seq)
#         #print("yo", norm_seq.shape)
#         coeff = dct(norm_seq, axis=1, norm='ortho')
#         q = np.around(coeff * self.scale).astype(int)
#         #print(q, norm_seq)

#         # Encode each sequence into BPE tokens
#         #self.bpe.enable_padding(length=319, pad_id=50000, pad_token="<PAD>")
#         #tokenizer.enable_truncation(max_length=319)
#         tokens: list[list[int]] = []
#         for b in range(batch_size):
#             flat = (q[b].flatten() - self.min_token +1).clip(min=0)
#             # chars = ''.join(map(chr, flat))
#             # print(b, flat, flat.shape)
#             # encoding = self.bpe.encode(chars)
#             # tokens.append(encoding.ids)
#             tokens.append(flat)

#         # apply truncation
#         # if truncation and max_length is not None:
#         #     tokens = [t[:max_length] for t in tokens]

#         # # apply padding
#         # self.pad_token_id = self.bpe.token_to_id("<PAD>")
#         # if padding and max_length is not None:
#         #     tokens = [
#         #         t + [self.pad_token_id] * (max_length - len(t))
#         #         if len(t) < max_length else t
#         #         for t in tokens
#         #     ]

#         # return tensors if requested
#         if return_tensors == 'pt':
#             # prepare input_ids and attention_mask
#             if max_length is None:
#                 max_length = max(len(t) for t in tokens)
#             #print([len(i) for i in tokens])
#             input_ids = torch.tensor(tokens, dtype=torch.long)
#             # attention mask: 1 for real tokens, 0 for padding
#             attention_mask = torch.tensor(
#                 [[1] * min(len(t), max_length) + [0] * max(0, max_length - len(t)) for t in tokens],
#                 dtype=torch.long
#             )
#             return {'input_ids': input_ids, 'attention_mask': attention_mask}

#         return tokens, T

#     def decode(
#         self,
#         token_ids: list[list[int]],
#         *,
#         time_horizon: int | None = None,
#         state_dim: int | None = None,
#     ) -> np.ndarray:
#         T = time_horizon or self.called_time_horizon
#         D = state_dim or self.called_state_dim
#         if T is None or D is None:
#             raise ValueError("Missing time_horizon or state_dim for decoding.")
        
#         decoded_seqs = []
#         for ids in token_ids:
#             # ids = np.abs(np.array(ids))
#             # print(ids)
#             # print([type(x) for x in ids])
#             #print(ids)
#             # txt = self.bpe.decode(ids) 
#             # arr = np.array(list(map(ord, txt))) 
#             # print(arr.shape, len(txt), len(ids))
#             #print(ids)
#             arr = ids + self.min_token -1
#             arr = arr.reshape(-1, D)
#             coeff = arr / self.scale
#             rec = idct(coeff, axis=0, norm='ortho')
#             rec = self._denormalize(rec)
#             decoded_seqs.append(rec)
#         return np.stack(decoded_seqs)

#     @classmethod
#     def fit_from_h5(
#         cls,
#         traj_path: str,
#         num_episodes: int = 5,
#         scale: float = 10.0,
#         vocab_size: int = 1000,
#         normalization: str = "zscore"
#     ) -> "TemporalBPEProcessor":
#         h5f = h5py.File(traj_path, 'r')
#         json_path = traj_path.replace('.h5', '.json')
#         json_data = io_utils.load_json(json_path)

#         seqs = []
#         if num_episodes == -1:
#             num_episodes = len(json_data['episodes'])
#         for ix, ep in enumerate(json_data['episodes'][:num_episodes]):
#             env_states = trajectory_utils.dict_to_list_of_dicts(
#                 h5f[f'traj_{ix}']['env_states']
#             )
#             seq = np.stack([
#                 s['articulations']['floating_gripper_v1']
#                 for s in env_states
#             ])
#             seqs.append(seq)

#         all_states = np.concatenate(seqs, axis=0)
#         mean = all_states.mean(axis=0)
#         #print(mean.shape)
#         std = all_states.std(axis=0)
#         std[std <= 1e-5] = 1.0
#         min_val = all_states.min(axis=0)
#         max_val = all_states.max(axis=0)

#         if normalization == "zscore":
#             normed_seqs = [(s - mean) / std for s in seqs]
#         elif normalization == "minmax":
#             normed_seqs = [(s - min_val) / (max_val - min_val + 1e-8) for s in seqs]
#         else:
#             raise ValueError("Normalization must be 'zscore' or 'minmax'.")

#         all_coeffs = [dct(s, axis=0, norm='ortho').flatten() for s in normed_seqs]
#         scaled_vals = np.around(np.concatenate(all_coeffs) * scale)
#         min_token = int(scaled_vals.min())
#         max_token = int(scaled_vals.max())

#         def token_iter():
#             for coeffs in all_coeffs:
#                 arr = (np.around(coeffs * scale) - min_token).astype(int)
#                 yield ''.join(map(chr, arr))

#         bpe = ByteLevelBPETokenizer()
#         alphabet = [chr(i) for i in range(max_token - min_token + 1)]
#         trainer = BpeTrainer(
#             vocab_size=vocab_size,
#             min_frequency=2,
#             initial_alphabet=alphabet,
#             special_tokens = ["<MASK>", "<PAD>"]
#         )
#        # bpe.add_special_tokens(["<MASK>", "<PAD>"])
#         bpe._tokenizer.train_from_iterator(token_iter(), trainer=trainer)
#         #bpe.enable_padding(length=319, pad_id=bpe.token_to_id("<PAD>"), pad_token="<PAD>")#pad_id=15000, pad_token="<PAD>")
        
# #         bpe.add_special_tokens({
# #     "pad_token": "[PAD]"
# # })
        
#         T, D = seqs[0].shape
#         return cls(
#             bpe_tokenizer=bpe,
#             scale=scale,
#             vocab_size=vocab_size,
#             min_token=min_token,
#             time_horizon=T,
#             state_dim=D,
#             normalization=normalization,
#             mean=mean,
#             std=std,
#             min_val=min_val,
#             max_val=max_val,
#         )

#     def save(self, save_dir: str):
#         os.makedirs(save_dir, exist_ok=True)
#         self.bpe.save_model(save_dir)
#         np.savez(
#             os.path.join(save_dir, "normalization_stats.npz"),
#             mean=self.mean,
#             std=self.std,
#             min_val=self.min_val,
#             max_val=self.max_val,
#             scale=self.scale,
#             min_token=self.min_token,
#             time_horizon=self.called_time_horizon,
#             state_dim=self.called_state_dim,
#             normalization=self.normalization,
#         )
#     def token_to_id(self, t):
#         return self.bpe.token_to_id(t)

#     @classmethod
#     def load(cls, save_dir) -> "TemporalBPEProcessor":
#         tokenizer_path = save_dir
#         stats_path = os.path.join(save_dir, "normalization_stats.npz")
#         bpe = ByteLevelBPETokenizer.from_file(
#             os.path.join(tokenizer_path, "vocab.json"),
#             os.path.join(tokenizer_path, "merges.txt"),
#         )
#         # bpe.add_special_tokens(["<MASK>", "<PAD>"])
#         # bpe.enable_padding(length=319, pad_id=bpe.token_to_id("<PAD>"), pad_token="<PAD>")
#         #print(bpe.token_to_id("<PAD>"))
        
#         stats = np.load(stats_path, allow_pickle=True)
#         return cls(
#             bpe_tokenizer=bpe,
#             scale=float(stats["scale"]),
#             vocab_size=None,
#             min_token=int(stats["min_token"]),
#             time_horizon=int(stats["time_horizon"]),
#             state_dim=int(stats["state_dim"]),
#             normalization=str(stats["normalization"]),
#             mean=stats["mean"],
#             std=stats["std"],
#             min_val=stats["min_val"],
#             max_val=stats["max_val"],
#         )

#     def plot_coeff_histogram(self, traj_path: str, episode_index: int, bins: int = 50):
#         h5f = h5py.File(traj_path, 'r')
#         json_path = traj_path.replace('.h5', '.json')
#         json_data = io_utils.load_json(json_path)
#         env_states = trajectory_utils.dict_to_list_of_dicts(
#             h5f[f'traj_{episode_index}']['env_states']
#         )
#         seq = np.stack([
#             s['articulations']['floating_gripper_v1']
#             for s in env_states
#         ])
#         norm_seq = self._normalize(seq)
#         coeffs = dct(norm_seq, axis=0, norm='ortho').flatten()
#         plt.figure(figsize=(8, 4))
#         plt.hist(coeffs, bins=bins)
#         plt.title(f"Normalized DCT Coeffs Histogram — Episode {episode_index}")
#         plt.xlabel("Normalized Coefficient")
#         plt.ylabel("Count")
#         plt.grid(True)
#         plt.show()

# def train_tokeniser(traj_path= 'monster_row_wise/2025_02_28_18_27_44_PickAnything.h5'):
#     tokeniser = TemporalBPEProcessor.fit_from_h5(
#         traj_path, num_episodes=-1, vocab_size=150, normalization="zscore"
#     )
#     tokeniser.save("saved_tokenizer_pickanything")

# def eval_tokeniser(tokeniser, traj_path= 'monster_row_wise/2025_02_28_18_27_44_PickAnything.h5'):
#     h5f = h5py.File(traj_path, 'r')
#     real_seq = []
#     for i in range(8):
#         env_states = trajectory_utils.dict_to_list_of_dicts(h5f['traj_1']['env_states'])
#         real_seq.append(np.stack([
#             s['articulations']['floating_gripper_v1']
#             for x,s in enumerate(env_states) if x<16
#         ]))
#     real_seq = np.array(real_seq)
#     print(real_seq.shape)
#     tokens, t = tokeniser(real_seq, padding=True, truncation= True, max_length=700)#tokeniser(real_seq)
#     print([len(i) for i in tokens] )
#     recon_seq = tokeniser.decode(tokens, time_horizon=t)[0]
#     ss_res = ((real_seq - recon_seq) ** 2).sum()
#     ss_tot = ((real_seq - real_seq.mean()) ** 2).sum()
#     r2_score = 1 - (ss_res / ss_tot)
#     print(f"R² Score: {r2_score:.6f}")
import os
import json
import numpy as np
import torch
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

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
        print(state_seq.shape)
        batch_size, T, D = state_seq.shape
        self.called_time_horizon = T
        self.called_state_dim = D

        norm_seq = self.normalize(state_seq)
        print('yo2',norm_seq.shape)
        coeff = dct(norm_seq, axis=1, norm='ortho')
        q = np.around(coeff * self.scale).astype(int)

        tokens: list[list[int]] = []
        for b in range(batch_size):
            flat = (q[b].flatten() - self.min_token).clip(min=0)
            tokens.append(flat.tolist())

        if return_tensors == 'pt':
            if max_length is None:
                max_length = max(len(t) for t in tokens)
            input_ids = torch.tensor(
                [t[:max_length] + [0] * max(0, max_length - len(t)) for t in tokens],
                dtype=torch.long
            )
            attention_mask = torch.tensor(
                [[1] * min(len(t), max_length) + [0] * max(0, max_length - len(t)) for t in tokens],
                dtype=torch.long
            )
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        #print(tokens)

        return torch.tensor(tokens)+1

    def decode(
        self,
        token_ids: list[list[int]],
        state_dim: int | None = None,
    ) -> np.ndarray:
        D = state_dim or self.called_state_dim
        token_ids = (token_ids-1).clip(min = 0)
        decoded_seqs = []
        print('min',self.min_token)
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
        pose_seqs = mat_to_pose10d(pose_seqs.reshape(-1,4,4))
        all_states = np.concatenate([pose_seqs, grip_seqs], axis= -1)
        print(len(all_states))
        mean = all_states.mean(axis=0)
        std = all_states.std(axis=0)
        std[std <= 1e-5] = 1.0
        min_val = all_states.min(axis=0)
        max_val = all_states.max(axis=0)
        #print(all_states.shape)
        # if normalization == "zscore":
        #     normed_seqs = [(s - mean) / std for s in seqs]
        # else:
        #     normed_seqs = [(s - min_val) / (max_val - min_val + 1e-8) for s in seqs]

        # # Determine token range without BPE
        # all_coeffs = [dct(s, axis=0, norm='ortho').flatten() for s in normed_seqs]
        # scaled_vals = np.around(np.concatenate(all_coeffs) * scale)
        min_token = 0#int(scaled_vals.min())

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

def train_tokeniser(data_dir: str = 'out_dataset_bottle', T=32):
    processor = TemporalBPEProcessor.fit_from_npz(
        data_dir, num_episodes=-1, scale=10, normalization="zscore", T=T
    )
    processor.save("saved_processor")


def eval_tokeniser(processor: TemporalBPEProcessor, data_dir: str = 'out_dataset', num_samples: int = 8):
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    real_seq = []
    ep = meta['episodes'][0]
    arr = np.load(os.path.join(data_dir, ep['file']))
    pose_seq = arr['pose']
    grip_seq = arr['grip'][..., None]
    for j in range(num_samples):
        i = np.random.randint(0, len(pose_seq)-processor.T)
        seq = np.concatenate([pose_seq[i:i+processor.T], grip_seq[i:i+processor.T]], axis=1)
        real_seq.append(seq)
    real_seq = np.array(real_seq)
    print(real_seq.shape)
    tokens= processor(real_seq)
    print(tokens)
    recon_seq = processor.decode(tokens)[0]
    ss_res = ((real_seq - recon_seq) ** 2).sum()
    ss_tot = ((real_seq - real_seq.mean()) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot)
    print(f"R² Score: {r2_score:.6f}")
