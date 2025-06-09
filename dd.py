# maniskill_data_utils.py
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

from train_ga import ManiSkillSequenceDataset, TemporalBPEProcessor
def get_relevant_info(ori_env, initial_pose: np.ndarray):
    """
    Extracts and preprocesses wrist-camera image, relative TCP pose, and gripper state.
    """
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
    """
    Plays back each episode, collects (img, pose, gripper) at each timestep t,
    and writes per-episode .npz shards plus meta.json in output_dir.
    The DataLoader will use these inputs to fetch the next state via the original HDF5.
    """
    os.makedirs(output_dir, exist_ok=True)
    episodes_dir = os.path.join(output_dir, 'episodes')
    os.makedirs(episodes_dir, exist_ok=True)

    # open source trajectory and metadata
    h5f = h5py.File(traj_path, 'r')
    with open(json_path, 'r') as f:
        meta = json.load(f)

    env_info = meta['env_info']
    episodes_meta = []
    env_info['env_kwargs']['render_mode']= "rgb_array"
    print(env_info['env_id'], env_info['env_kwargs'])
    env = gym.make(env_info['env_id'], **env_info['env_kwargs'])
    for ix, ep in enumerate(meta['episodes']):
        # if ix==1:
        #     break
        env.reset(**ep['reset_kwargs'])
        states = trajectory_utils.dict_to_list_of_dicts(h5f[f"traj_{ix}/env_states"])
        length = ep['episode_len']

        imgs, poses, grips = [], [], []
        for t in range(length - 2):
            env.set_state_dict(states[t])
            i, p, g = get_relevant_info(env, initial_pose)
            print(p.shape)
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


class ManiSkillSequenceDataset2(Dataset):
    """
    Loads the prebuilt .npz shards for inputs and fetches the next state dict from the
    original HDF5. Returns:
      {
        'obs': {'image': Tensor[C,H,W], 'pose': Tensor[16], 'gripper': Tensor[1]},
        'target_state': dict  # raw state_dict for env.set_state_dict()
      }
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # load metadata
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.traj_path = meta['traj_path']
        self.initial_pose = np.array(meta['initial_pose'], dtype=np.float32)
        self.episodes = meta['episodes'][0]
        # open HDF5 for state lookup
        self.h5 = h5py.File(self.traj_path, 'r')
        # preload states per episode
        self.states = []
        for ep in self.episodes:
            print(f"traj_{ep['index']}/env_states")
            grp = self.h5[f"traj_{ep['index']}/env_states"]
            self.states.append(trajectory_utils.dict_to_list_of_dicts(grp))

        # build cumulative index for dataset length
        lengths = [ep['length'] for ep in self.episodes]
        self.cumlen = np.cumsum([0] + lengths)
        print(self.cumlen)

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx: int):
        # map global idx -> (episode, step)
        ep = int(np.searchsorted(self.cumlen, idx, side='right') - 1)
        step = idx - self.cumlen[ep]
        # load input arrays
        arr = np.load(os.path.join(self.data_dir, self.episodes[ep]['file']))
        img = arr['img'][step]
        pose = arr['pose'][step]
        grip = arr['grip'][step]

        # apply transform or default to [C,H,W] float
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2,0,1).float().div(255.)

        # prepare tensors
        pose = torch.from_numpy(pose).float()
        grip = torch.tensor([grip], dtype=torch.float32)

        # fetch raw next state dict
        target_state = self.states[ep][step + 1]

        return {
            'obs': {'image': img, 'pose': pose, 'gripper': grip},
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
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
tokeniser = tokeniser = TemporalBPEProcessor.load("saved_processor")
for batch in loader:
    p = batch['target_state']#np.concatenate(batch['target_state'])#, axis=-1)  # shape: [3, 224, 224](
    print(p.shape)
    x = tokeniser(p)
    for i in x:
      print(i)
    # img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # # Convert from RGB to BGR
    # #img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # # Show image
    # cv2.imshow("Sample Image", img_np)
    # cv2.waitKey(1)
    # print(batch['obs']['image'].shape)
