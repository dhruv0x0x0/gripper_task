import torch
import torch.optim as optim
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import mat_to_pose9d
class Action_Dataset(Dataset):
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
        all_poses = mat_to_pose9d(arr['pose'].reshape(-1, 4, 4))
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
    
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class Video_Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None, num_past_frames=1, num_fut_frames=1):
        self.data_dir = data_dir
        self.transform = transform
        self.num_past_frames = num_past_frames
        self.num_fut_frames = num_fut_frames

        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.episodes = meta['episodes']
        lengths = [ep['length'] for ep in self.episodes]
        self.cumlen = np.cumsum([0] + lengths)
        self.current_ep = -1
        self.arr = None

    def __len__(self):
        return int(self.cumlen[-1])

    def _get_frames(self, imgs, indices):
        num_frames = len(indices)
        h, w, c = imgs[0].shape
        frames = torch.empty((num_frames, c, h, w), dtype=torch.float32)

        for i, idx in enumerate(indices):
            img = imgs[idx]
            img_t = self.transform(img) if self.transform else torch.from_numpy(img).permute(2, 0, 1).float().div(255.)
            frames[i] = img_t
        return frames

    def __getitem__(self, idx: int):
        ep = int(np.searchsorted(self.cumlen, idx, side='right') - 1)
        step = idx - self.cumlen[ep]

        if ep != self.current_ep:
            self.current_ep = ep
            self.arr = np.load(os.path.join(self.data_dir, self.episodes[ep]['file']), mmap_mode='r')

        imgs = self.arr['img']
        episode_len = imgs.shape[0]

        past_indices = [max(step - i, 0) for i in range(self.num_past_frames, 0, -1)]
        future_indices = [min(step + i, episode_len - 1) for i in range(self.num_fut_frames)]

        past_frames = self._get_frames(imgs, past_indices)
        future_frames = self._get_frames(imgs, future_indices)

        return {
            'past_frames': past_frames,
            'future_frames': future_frames
        }


class Image_Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.episodes = meta['episodes']
        self.lengths = [ep['length'] for ep in self.episodes]
        self.cumlen = np.cumsum([0] + self.lengths)
        self.current_ep = -1
        self.arr = None

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx: int):
        ep = int(np.searchsorted(self.cumlen, idx, side='right') - 1)
        step = idx - self.cumlen[ep]

        if ep != self.current_ep:
            self.current_ep = ep
            self.arr = np.load(os.path.join(self.data_dir, self.episodes[ep]['file']), mmap_mode='r')

        img = self.arr['img'][step]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.)

        return img

  
if __name__ == '__main__':
    from torchvision import transforms
    #ds =  Video_Dataset('out_dataset_bottle', transform=transforms.ToTensor())
    ds2 =  Image_Dataset('out_dataset_bottle', transform=transforms.ToTensor())
    loader = DataLoader(
        ds2,
        batch_size=64,
        shuffle=False,
        num_workers=8,       # try increasing
        pin_memory=True,     # pin memory for faster .cuda() transfers
        prefetch_factor=2    # each worker can prefetch up to 2 batches
    )
    for batch in loader:
        p = batch#batch['past_frames']
        print(p.shape)
        for img_tensor in p:
            #img_tensor = img_tensor.squeeze(0)
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imshow("Sample Image", img_np)
            cv2.waitKey(10)
