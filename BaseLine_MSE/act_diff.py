from torchvision import transforms
import h5py
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
import time
import sim_env.maniskill_env.envs
import cv2
import numpy as np
import sapien
import gymnasium as gym
import cv2
import sapien
from utils import pose9d_to_mat, mat_to_pose9d
import torch
action_list = []
from utils import StateQueue_dict
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import h5py
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
import time
import sim_env.maniskill_env.envs
import cv2
from diff_policy import DiffusionPolicy
action_list = []
vid = []
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_action(offset_pose: np.ndarray=np.eye(4), gripper_state: float=1) -> np.ndarray:
    initial_pose = np.array(
        [
            [1, 0, 0.1, 0.097],
            [0, 1.0, 0.0, 0.0],
            [-0.1, 0.0, 1, 0.378],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_pose = initial_pose @ offset_pose 
    pose = sapien.Pose(target_pose)
    action = np.zeros(8)
    action[:3] = pose.get_p()
    action[3:7] = pose.get_q()
    action[7] = gripper_state
    return action


class SimModelRunner:
    def __init__(self, s, actions, env: gym.Env, initial_pose: np.ndarray, device, writer=None):
        self.env = env
        self.initial_pose = initial_pose
        self.device = device
        self.actions = actions
        self.writer = writer

        # warm up steps
        # for i in range(50):
        #     obs, _, _, _, _ = self.env.step(get_action())
        self.buffer = StateQueue_dict(k=2)
        for i in range(100):
            #self.env.step(get_action())
            self.env.set_state_dict(s[i])
            self.env.render()
            self.buffer.update(self.get_current_obs())
        #self.env.step(get_action())

        # self.env.set_state_dict(s[2])
        # self.env.render()
        # self.buffer.update(self.get_current_obs())

        self.preprocess = transforms.Compose([transforms.ToTensor()])
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        self.diff_policy = DiffusionPolicy(noise_scheduler=scheduler, load_path='modeltry_ema_epoch14.pt')

    def get_current_obs(self):
        img_np, relT, grip = get_relevant_info(self.env, self.initial_pose)
        vid.append(img_np)
        
        img_t = transforms.ToTensor()(img_np).to(device=self.device)
        return {
            'pose': mat_to_pose9d(relT),
            'gripper': np.array([grip]),
            'image': img_t
        }

    @torch.no_grad()
    def step(self, j, play_rec):
        if play_rec:
            for i in range(16):
                action = self.actions[j*16 + i]
                print(action)
                obs, _, _, _, info = self.env.step(action)
                frame = self.env.render()
                frame = frame[0]
                if self.writer:
                    # convert RGB to BGR for OpenCV
                    self.writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        else:
            next_action = self.diff_policy.predict_action(obs=self.buffer.get())
            next_action = np.array(next_action['action_pred'][0].cpu())

            poses = next_action[:, :9]
            poses = pose9d_to_mat(poses)
            gripper = next_action[:, 9:]
            
            for i in range(16):
                action = get_action(poses[i], 2*(gripper[i]-0.55))
                print(action)
                action_list.append(action)
                obs, _, _, _, info = self.env.step(action)
                frame = self.env.render()
                frame = frame[0]
                if self.writer:
                    self.writer.write(cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR))
                self.buffer.update(self.get_current_obs())

        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default="monster_row_wise/2025_02_28_18_27_44_PickAnything.h5")
    parser.add_argument('--episode', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--play_recorded', type=int, default=0)

    args = parser.parse_args()

    traj_path = args.h5_path
    ori_h5_file = h5py.File(traj_path, "r")
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    json_data['env_info']["env_kwargs"]['show_goal_site']= False
    env_info = json_data["env_info"]
    print(env_info)
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    ori_env_kwargs['render_mode'] = 'rgb_array'
    ori_env = gym.make(env_id, **ori_env_kwargs)
    ori_env.reset(**json_data['episodes'][args.episode]['reset_kwargs'])
    ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    ori_h5_file[f"traj_{args.episode}"]["env_states"]
                    )
    ori_env.set_state_dict(ori_env_states[0])
    ori_env.render()
    
    # ——— VIDEO RECORDER SETUP ———
    first_rgb = ori_env.render()
    #print(first_rgb.shape)
    first_rgb = first_rgb[0]
    h, w, _ = first_rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # adjust to match your simulation speed
    video_path = 'sim_record.mp4'
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    # ————————————————————————

    initial_pose = np.array(
        [
            [1, 0, 0, 0.097],
            [0, 1, 0, 0],
            [-0.1, 0, 1, 0.378],
            [0, 0, 0, 1]
        ],
        dtype=np.float32
    )
    #first_frame, _, _ = get_relevant_info(ori_env=ori_env, initial_pose=initial_pose)
    actions = np.load('actions.npy')

    runner = SimModelRunner(ori_env_states, actions, ori_env, initial_pose, device, writer)

    for step in range(args.max_steps):
        runner.step(j=step, play_rec=args.play_recorded)

    # finish recording
    writer.release()
    print(f"Saved simulation video to {video_path}")
    if len(action_list)>0:
         all_actions = np.stack(action_list, axis=0)   # shape: [T, action_dim]
         np.save("actions.npy", all_actions)
    # vid2 = np.stack(vid, axis=0)
    # np.save("vid.npy", vid2)
    ori_env.close()


if __name__ == '__main__':
    main()
