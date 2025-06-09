import torch
import torch.optim as optim
import os
from ema import EMAModel
from diffusers.optimization import get_scheduler
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from diff_policy import DiffusionPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils import mat_to_pose9d

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
    

def train(dataloader, epochs: int = 100, save_path: str = 'model'): 
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer= optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * epochs
        )

        ema = EMAModel(diff_policy.model, update_after_step=0, inv_gamma=1.0, power=2/3)

        epoch_losses = []
        print('training started')

        for epoch in range(epochs):
            total_loss = 0.0
            for i, batch in enumerate(dataloader):
                loss = diff_policy.compute_loss(batch)
                if i % 10 == 0:
                    print(f"Batch {i} loss: {loss.item():.4f}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                ema.step(diff_policy.model)

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            torch.save(ema.averaged_model.state_dict(), f"{save_path}try_ema_epoch{epoch+1}.pt")


from torchvision import transforms
if  __name__== '__main__':
    mode = 'train'
    scheduler = DDPMScheduler(num_train_timesteps= 100, beta_schedule="squaredcos_cap_v2")
    diff_policy = DiffusionPolicy(noise_scheduler= scheduler)
    optimizer = optim.AdamW(diff_policy.model.parameters(), lr=1e-4, weight_decay=1e-3, betas=[0.9,0.95])
    batch_size = 64
    epochs = 50
    #tokeniser_path = "saved_processor" 
    print('loading_dataset')
    ds = ManiSkillSequenceDataset('out_dataset_bottle', transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    train(loader)