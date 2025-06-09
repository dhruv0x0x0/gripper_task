import os
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from mani_skill.utils import io_utils
from mani_skill.trajectory import utils as trajectory_utils
import gymnasium as gym
import sim_env.maniskill_env.envs 
import sapien
import torch
from utils import TemporalBPEProcessor, mat_to_pose9d, pose9d_to_mat

class EpisodeDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 traj_json: str,
                 traj_h5: str,
                 episode_idx: int,
                 chunk_size: int = 4,
                 initial_pose: np.ndarray = None,
                 renderer_mode: str = 'rgb_array',
                 tokenizer_path: str = 'saved_processor'):
        
        self.chunk_size = chunk_size
        self.initial_pose = initial_pose
        # load pose+grip data
        ep_file = os.path.join(data_dir, 'episodes', f'ep_{episode_idx}.npz')
        data = np.load(ep_file)
        poses_flat = data['pose']  # (T,16)
        grips = data['grip']  # (T,)
        mats = poses_flat.reshape(-1, 4, 4)
        pose10d = mat_to_pose9d(mats)  # (T,9)

        # load tokenizer
        self.tokenizer = TemporalBPEProcessor.load(tokenizer_path)

        # load environment
        meta = io_utils.load_json(traj_json)
        meta['env_info']["env_kwargs"]['show_goal_site']= False
        ori_h5 = h5py.File(traj_h5, 'r')
        env_states = trajectory_utils.dict_to_list_of_dicts(
            ori_h5[f"traj_{episode_idx}"]["env_states"]
        )
        env_info = meta['env_info']
        env_info['env_kwargs']['render_mode'] = renderer_mode
        self.env = gym.make(env_info['env_id'], **env_info['env_kwargs'])
        episode_meta = meta['episodes'][episode_idx]
        self.env.reset(**episode_meta['reset_kwargs'])
        self.env.set_state_dict(env_states[0])

        # prepare samples
        self.samples = []
        T = pose10d.shape[0]
        for start in range(0, T, chunk_size):
            end = start + chunk_size
            if end > T:
                break
            # extract chunk
            chunk_pose = pose10d[start:end]  # (chunk,9)
            chunk_grip = grips[start:end].reshape(-1,1)
            # encode tokens
            tok = np.concatenate([chunk_pose, chunk_grip], axis=-1).reshape(1, -1, 10)
            enc = self.tokenizer(tok).numpy()  
            if start == 0:   
                print(enc[:,:10])      
                enc[:,:10] = np.array([38, 68, 134,  88, 114, 141,  60, 106, 103, 111]) #+ tokenizer.min_token

            enc[:,10:] = 1- self.tokenizer.min_token
            img1 = self._get_rgb()
            next_action = self.tokenizer.decode(torch.tensor(enc))[0]    
            poses =  next_action[:, :9] 
            gripper = next_action[:, 9:]  #time.sleep(0.1)
            poses =  pose9d_to_mat(poses)

            # step through and collect future frames
            future_frames = []
            for p_vec, g_val in zip(poses, gripper):
                action_vec = self._to_action(p_vec, 2*(g_val-0.57))
                self.env.step(action_vec)
                img2 = self._get_rgb()
                future_frames.append(img2)
            # stack
            future_stack = np.stack(future_frames, axis=0)  # (chunk,H,W,3)

            self.samples.append({
                'current_frame': img1,             # (H,W,3)
                'future_frames': future_stack,     # (chunk,H,W,3)
                'action': enc[:,:10][0]                   # (chunk,10)
            })

    def _get_rgb(self):
        img = self.env.get_sensor_images()['wrist_cam']['rgb'][0].cpu().numpy()
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # pad/crop/resize as needed
        new_f = np.zeros((400,400,3), np.uint8)
        pad = (400 - 224) // 2
        new_f[pad:pad+224] = frame
        new_f = cv2.resize(new_f, (224,224), interpolation=cv2.INTER_AREA)
        # convert to tensor
        return torch.from_numpy(new_f).permute(2,0,1).float() / 255.0

    def _to_action(self, pose10d: np.ndarray, grip: np.ndarray):
        target = self.initial_pose @ pose10d
        p = sapien.Pose(target)
        a = np.zeros(8, dtype=np.float32)
        a[:3] = p.get_p()
        a[3:7] = p.get_q()
        a[7] = grip
        return a

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'current_frame': s['current_frame'],
            'future_frames': s['future_frames'],
            'action': torch.from_numpy(s['action']).float()
        }

if __name__ == '__main__':
    initial_pose = np.array([
        [1,0,0,0.097],
        [0,1,0,0],
        [-0.1,0,1,0.378],
        [0,0,0,1]
    ], dtype=np.float32)

    dataset = EpisodeDataset(
        data_dir='out_dataset_bottle',
        traj_json='monster_row_wise/2025_02_28_18_27_44_PickAnything.json',
        traj_h5='monster_row_wise/2025_02_28_18_27_44_PickAnything.h5',
        episode_idx=4,
        chunk_size=4,
        initial_pose=initial_pose
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for batch in loader:
        print(batch['current_frame'].shape)   # (B,3,224,224)
        print(batch['future_frames'].shape)   # (B,chunk,3,224,224)
        print(batch['action'])          # (B,chunk,10)
        for chunk in batch['future_frames']:
             #for img in chunk:
                img = chunk[0]
                img_np = img.mul(255).byte().cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                cv2.imshow("Image", img_np)
                cv2.waitKey(500)
    
