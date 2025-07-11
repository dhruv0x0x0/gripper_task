import os
import json
import h5py
import numpy as np
import gymnasium as gym
from mani_skill.trajectory import utils as trajectory_utils
import sim_env.maniskill_env.envs  # register custom envs
import cv2
import sapien
from torch.utils.data import Dataset, DataLoader
import torch
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

def get_relevant_info(ori_env, initial_pose: np.ndarray):
    obs = ori_env.get_obs()
    imgs = ori_env.get_sensor_images()
    wrist = imgs['wrist_cam']['rgb'][0].cpu().numpy()
    frame = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    pad = (400 - 224) // 2
    canvas[pad:pad+224, :, :] = frame
    canvas = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA)

    tcp = obs['extra']['tcp_pose'][0]
    grip = obs['agent']['qpos'][0, 7] / 0.04
    T = sapien.Pose(p=tcp[:3], q=tcp[3:]).to_transformation_matrix()
    relT = np.linalg.inv(initial_pose) @ T

    return canvas, relT.astype(np.float32), np.float32(grip)


def build_dataset(traj_path: str,
                  json_path: str,
                  output_dir: str,
                  initial_pose: np.ndarray):
    
    os.makedirs(output_dir, exist_ok=True)
    episodes_dir = os.path.join(output_dir, 'episodes')
    os.makedirs(episodes_dir, exist_ok=True)
    h5f = h5py.File(traj_path, 'r')
    with open(json_path, 'r') as f:
        meta = json.load(f)
    meta['env_info']["env_kwargs"]['show_goal_site']= False

    env_info = meta['env_info']
    episodes_meta = []
    env_info['env_kwargs']['render_mode']= "rgb_array"
    #print(env_info['env_id'], env_info['env_kwargs'])
    env = gym.make(env_info['env_id'], **env_info['env_kwargs'])
    for ix, ep in enumerate(meta['episodes']):
        env.reset(**ep['reset_kwargs'])
        states = trajectory_utils.dict_to_list_of_dicts(h5f[f"traj_{ix}/env_states"])
        length = ep['episode_len']
        imgs, poses, grips = [], [], []
        for t in range(length - 2):
            env.set_state_dict(states[t])
            i, p, g = get_relevant_info(env, initial_pose)
            #print(p.shape)
            imgs.append(i)
            poses.append(p.reshape(-1))
            grips.append(g)
            env.render()

        arrs = {
            'img': np.stack(imgs),
            'pose': np.stack(poses),
            'grip': np.stack(grips),
        }
        shard_path = os.path.join(episodes_dir, f'ep_{ix}.npz')
        np.savez_compressed(shard_path, **arrs)
        episodes_meta.append({
            'index': ix,
            'length': length - 3,
            'file': f'episodes/ep_{ix}.npz'
        })
        print(f"Saved episode {ix}, steps={length - 1}")

    # write overall metadata, including reference to the source HDF5
    out_meta = {
        'traj_path': traj_path,
        'env_info': env_info,
        'initial_pose': initial_pose.tolist(),
        'episodes': episodes_meta
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(out_meta, f, indent=2)
    print(f"Dataset built at {output_dir}")


class ManiSkillSequenceDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, state_horizon: int = 16, past_poses: int = 5):
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


traj_path = "raw_data/bottle_rows/2025_03_04_13_51_38_PickAnything.h5"
json_path = traj_path.replace(".h5", ".json")
init_pose = np.array([
    [1,0,0,0.097],
    [0,1,0,0],
    [-0.1,0,1,0.378],
    [0,0,0,1]

], dtype=np.float32)
#build_dataset(traj_path, json_path, 'out_dataset_bottle', init_pose)
from torchvision import transforms
ds = ManiSkillSequenceDataset('out_dataset_bottle', transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
for batch in loader:
    img = batch['obs']['image'][0]
    for img_tensor in img:
      # shape: [3, 224, 224]
        print(img_tensor.shape)
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Convert from RGB to BGR
        #img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Show image
        cv2.imshow("Sample Image", img_np)
        cv2.waitKey(500)
        print(batch['obs']['image'].shape)
