import h5py
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
import time
import argparse
import sim_env.maniskill_env.envs
import os
import numpy as np
import sapien
np.set_printoptions(precision=3, suppress=True)
from train_ga import TemporalBPEProcessor, pose10d_to_mat, mat_to_pose10d, create_model, train_tokeniser
import torch
import torchvision

# Constants and paths
data_dir = 'out_dataset_bottle'
json_dir = 'monster_row_wise'
traj_ext = '.h5'
JSON_PATH = "monster_row_wise/2025_02_28_18_27_44_PickAnything.json"

# Initial pose used in preprocessing (must match build_dataset)
initial_pose = np.array([
    [1, 0, 0, 0.097],
    [0, 1, 0, 0.0],
    [-0.1, 0, 1, 0.378],
    [0, 0, 0, 1]
], dtype=np.float32)

def get_action(offset_pose: np.ndarray = np.eye(4), gripper_state: float = 1) -> np.ndarray:
    target_pose = initial_pose @ offset_pose
    pose = sapien.Pose(target_pose)
    action = np.zeros(8)
    action[:3] = pose.get_p()
    action[3:7] = pose.get_q()
    action[7] = gripper_state
    return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=8, help='sequence length')
    parser.add_argument('--o', type=int, default=0, help='overlap size')
    parser.add_argument('--num_eps', type=int, default=100 ,help='number of episodes to run')
    parser.add_argument('--out_txt', type=str, default='actions.txt', help='output txt file')
    args = parser.parse_args()

    T = args.T
    o = args.o
    chunk_size = T - o
    num_eps = args.num_eps

    # Train or load tokenizer
    train_tokeniser(T=T)
    tokenizer = TemporalBPEProcessor.load('saved_processor')

    all_actions = []  # to store arrays from all episodes
    xx = []
    for ep_idx in range(num_eps):
        # Load episode data
        ep_file = os.path.join(data_dir, 'episodes', f'ep_{ep_idx}.npz')
        data = np.load(ep_file)
        poses_flat = data['pose']
        grips = data['grip']
        mats = poses_flat.reshape(-1, 4, 4)
        pose10d = mat_to_pose10d(mats)

        # Load environment metadata
        json_path = JSON_PATH#os.path.join(json_dir, f"{json_dir}/{JSON_PATH.split('/')[-1]}")
        meta = io_utils.load_json(json_path)
        traj_path = "monster_row_wise/2025_02_28_18_27_44_PickAnything.h5"
        ori_h5_file = h5py.File(traj_path, 'r')

        ori_env_states = trajectory_utils.dict_to_list_of_dicts(
            ori_h5_file[f"traj_{ep_idx}"]["env_states"]
        )
        env_info = meta['env_info']
        env_info['env_kwargs']['render_mode'] = 'rgb_array'
        #env = gym.make(env_info['env_id'], **env_info['env_kwargs'])
        episode = meta['episodes'][ep_idx]
        #env.reset(**episode['reset_kwargs'])
       # env.set_state_dict(ori_env_states[0])

        # Process in chunks
        a = []
        num_steps = pose10d.shape[0]
        
        for start in range(0, num_steps, chunk_size):
            end = start + chunk_size + o
            if end > num_steps:
                break
            chunk_pose = pose10d[start:end]
            chunk_g = grips[start:end].reshape(-1, 1)
            tok = np.concatenate([chunk_pose, chunk_g], axis=-1).reshape(1, -1, 10)
            tokens = tokenizer(tok)
            tokens[:, 10:] = 1 - tokenizer.min_token
            next_action = tokenizer.decode(torch.tensor(tokens))[0]
            poses = next_action[:, :9]
            gripper = next_action[:, 9:]

            a.append(tokens[:, :10].numpy()[0])
            #print(f"Episode {ep_idx}, Chunk {start}:{end} - pose err: {np.mean(np.abs(poses - chunk_pose)):.6f}, grip err: {np.mean(np.abs(gripper - chunk_g)):.6f}")
        xx.append(a[0])
        #env.close()
        # Stack and offset
        episode_actions = np.stack(a).T + tokenizer.min_token
        
        all_actions.append(episode_actions)

    # Combine all episodes and save to txt
    xx = np.stack(xx, axis=0)
    print(xx.shape)
    with open(args.out_txt, 'w') as f:
        for ep_array in all_actions:
            #print(ep_array)
            # each row is 10 integers
            #f.write(ep_array)
            np.savetxt(f, ep_array, fmt='%d')
            f.write('\n-----------------\n')

    print(f"Saved actions of {num_eps} episodes to {args.out_txt}")
    print(xx)
    print(xx.mean(axis=0), xx.std(axis=0))
    print(np.round(xx.mean(axis=0)))