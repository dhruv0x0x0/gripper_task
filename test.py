import h5py
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
import time
import argparse
import sim_env.maniskill_env.envs
import os
#from einops import rearrange
import numpy as np
import sapien
np.set_printoptions(precision=3, suppress=True)
from train_ga import TemporalBPEProcessor, pose10d_to_mat, mat_to_pose10d, create_model, train_tokeniser
import torch
import torchvision
import cv2
# Constants and paths
DATA_DIR = 'out_dataset_bottle'
EP_IDX = 204
CHUNK_SIZE = 4
JSON_PATH = "monster_row_wise/2025_02_28_18_27_44_PickAnything.json"
traj_path = "monster_row_wise/2025_02_28_18_27_44_PickAnything.h5"
train_tokeniser(T = CHUNK_SIZE)


initial_pose = np.array([
    [1,0,0,0.097],
    [0,1,0,0],
    [-0.1,0,1,0.378],
    [0,0,0,1]
], dtype=np.float32)

def get_relevant_info(env):
    obs = env.get_obs()
    image_dict = env.get_sensor_images()
    wrist_cam = image_dict['wrist_cam']['rgb'][0].cpu().numpy()
    frame = cv2.cvtColor(wrist_cam,cv2.COLOR_RGB2BGR)
    #frame = cv2.resize(wrist_cam,(1280,720))
    new_frame = np.zeros((400,400,3),np.uint8)
    pad = int((400-224)/2)
    new_frame[pad:224+pad,:,:] = frame
    new_frame = cv2.resize(new_frame,(224,224),interpolation=cv2.INTER_AREA)
    tcp_pose = obs['extra']["tcp_pose"][0]
    gripper_state = obs['agent']["qpos"][0,7]/0.04
    tcp_pose_matrix = sapien.Pose(p=tcp_pose[:3],q=tcp_pose[3:]).to_transformation_matrix()
    rel_tcp_pose_matrix = np.linalg.inv(initial_pose) @ tcp_pose_matrix
    return new_frame, rel_tcp_pose_matrix,gripper_state

def get_action(offset_pose: np.ndarray=np.eye(4), gripper_state: float=1) -> np.ndarray:
    target_pose = initial_pose @ offset_pose 
    pose = sapien.Pose(target_pose)
    action = np.zeros(8)
    action[:3] = pose.get_p()
    action[3:7] = pose.get_q()
    action[7] = gripper_state
    return action


def main():
    tokenizer = TemporalBPEProcessor.load('saved_processor')
    ep_file = os.path.join(DATA_DIR, 'episodes', f'ep_{EP_IDX}.npz')
    data = np.load(ep_file)
    poses_flat = data['pose']  # shape: (T, 16)
    grips = data['grip']      
    mats = poses_flat.reshape(-1, 4, 4)
    pose10d = mat_to_pose10d(mats)  # shape
    meta = io_utils.load_json(JSON_PATH)
    ori_h5_file = h5py.File(traj_path, "r")
    ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    ori_h5_file[f"traj_{EP_IDX}"]["env_states"]
                    )
    env_info = meta['env_info']
    #env_info['env_kwargs']['render_mode'] = 'rgb_array'
    env = gym.make(env_info['env_id'], **env_info['env_kwargs'])
    episode = meta['episodes'][EP_IDX]
    env.reset(**episode["reset_kwargs"])
    env.set_state_dict(ori_env_states[0])
    # tokens = tokenizer(np.concatenate([pose10d, grips.reshape(-1,1)], axis=-1))[0].cpu().numpy()
    # print(list(tokens))
    # for i in tokens.cpu():
    #     print(i)

    #Process in chunks
    a = []
    num_steps = pose10d.shape[0]
    for start in range(0, num_steps, CHUNK_SIZE):
        if start+CHUNK_SIZE>num_steps:
            break
        end = start + CHUNK_SIZE 
        chunk_pose = pose10d[start:end] 
        chunk_g = grips[start:end].reshape(-1, 1)     
        tok = np.concatenate([chunk_pose, chunk_g], axis=-1).reshape(1, -1, 10)
        encoded_tokens = tokenizer(tok).numpy()
        if start == 0:   
            print(encoded_tokens[:,:10])      
            encoded_tokens[:,:10] = np.array([38, 68, 134,  88, 114, 141,  60, 106, 103, 111]) #+ tokenizer.min_token
        encoded_tokens[:,10:] = 1- tokenizer.min_token

        next_action = tokenizer.decode(torch.tensor(encoded_tokens))[0]
        
        poses =  next_action[:, :9] 
        gripper = next_action[:, 9:]  #time.sleep(0.1)

        a.append(encoded_tokens[:,:10])   # you should store this encoded_tokens[:,:10] for dataset
        print(f"Chunk {start}:{end} - pose10d max err: {np.mean(np.abs(poses - chunk_pose)):.6f}")
        print(f"Chunk {start}:{end} - grip    max err: {np.mean(np.abs(gripper- chunk_g)):.6f}")       
        poses =  pose10d_to_mat(poses)
        img, _, _ = get_relevant_info(env=env)  #get input image for model before executing the chunk
        for p_vec, g_val in zip(poses, gripper):
            action = get_action(p_vec, 2*(g_val-0.57))
            img, _, _ = get_relevant_info(env=env)  #after each step execution get the image
            env.step(action)
            env.render()
            #time.sleep(0.1)

    env.close()
    print(np.stack(a).T+tokenizer.min_token)

if __name__ == '__main__':
    main()
