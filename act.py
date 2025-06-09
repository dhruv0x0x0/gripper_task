#!/usr/bin/env python
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
import argparse
import sim_env.maniskill_env.envs
import cv2
import numpy as np
import sapien
import gymnasium as gym
import cv2
import sapien
from typing import Sequence
from policy import replace_bn_with_gn, generate
from train_ga import TemporalBPEProcessor, pose10d_to_mat, mat_to_pose10d, create_model
import torch
import torchvision
action_list = []
from utils import StateQueue
#!/usr/bin/env python
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

action_list = []

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    if weights and weights.lower() == "r3m":
        return get_r3m(name=name, **kwargs)
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def get_r3m(name, **kwargs):
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    resnet_model = model.module.convnet.to('cpu')
    return resnet_model

# ... [other functions unchanged] ...

def get_relevant_info(ori_env, initial_pose: np.ndarray):
    obs = ori_env.get_obs()
    imgs = ori_env.get_sensor_images()
    wrist = imgs['wrist_cam']['rgb'][0].cpu().numpy()
    frame = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

    canvas = np.zeros((450000, 400, 3), dtype=np.uint8)
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
    #print(np.rad2deg(pose.get_rpy()))
    #print(f"Input: {pose.get_p()}")
    action[:3] = pose.get_p()
    action[3:7] = pose.get_q()
    action[7] = gripper_state
    return action

class StateCodec:
    """
    Convert between raw env.states (dicts) and token sequences.
    You should adapt these methods to your own quantization / binning scheme.
    """
    def __init__(self, tokenizer: TemporalBPEProcessor, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def state_to_tokens(self, state: dict) -> torch.LongTensor:
        # Simple JSON → BPE tokens. You may want a more compact numeric encoding.
        state_str = json.dumps(state, sort_keys=True)
        tokens, _ = self.tokenizer([state_str])
        # tokens is List[List[int]]
        return torch.tensor(tokens[0], dtype=torch.long, device=device)

    def tokens_to_state(self, tokens: Sequence[int]) -> dict:
        state_str = self.tokenizer.decode(tokens)
        return state_str#json.loads(state_str)


class SimModelRunner:
    def __init__(self, actions, env: gym.Env, model: nn.ModuleDict,
                 tokenizer: TemporalBPEProcessor, state_codec: StateCodec,
                 initial_pose: np.ndarray, writer: cv2.VideoWriter, device: str = 'cpu'):
        self.env = env
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.codec = state_codec
        self.initial_pose = initial_pose
        self.device = device
        self.actions = actions
        self.buffer = StateQueue(k=2)
        self.buffer.update(self.get_current_obs())
        self.preprocess = transforms.Compose([transforms.ToTensor()])
        self.writer = writer
    def get_current_obs(self):
        img_np, relT, grip = get_relevant_info(self.env, self.initial_pose)
        gripper_state = np.concatenate([mat_to_pose10d(relT), np.array([grip])], axis= -1)
        img_np = torch.tensor(img_np).to(device= self.device)
        img_np = img_np.permute(2, 0, 1).float().div(255.)
        #print(img_np.dtype)
        image_features = self.model['vision_encoder'](img_np.view(-1, 3, 224, 224))
        B, C, H, W = image_features.shape
        image_features = image_features.reshape(B, C*H*W)#.permute(0, 2, 1) 
        obs_features = torch.cat([image_features.squeeze(0), torch.tensor(gripper_state).to(self.device)], dim=-1)
        return obs_features
    @torch.no_grad()
    def step(self,j):
        if 0:
            for i in range(16):
                action = self.actions[j*16+i]
                print(action)
                obs, _, _, _,info = self.env.step(action)
                self.env.render()
        else:
            n_obs = self.buffer.get()
            n_obs = n_obs.reshape(1,-1)
            input_ids = None
            new_tokens = generate(model =self.model['lldm'], prompt= input_ids, cond= n_obs, mask_id = 0, device= device)#logits = self.model['lldm'](input_ids, obs_feats)      # [1, seq_len, vocab]
            ones = torch.ones(150, dtype=new_tokens.dtype).to(self.device).reshape(1,-1)
            new_tokens = torch.cat([new_tokens, ones], dim=-1)
            next_action= self.codec.tokens_to_state(new_tokens.cpu()).squeeze(0)
            poses =  next_action[:, :9]   # first 9 columns
            poses =  pose10d_to_mat(poses)
            gripper = next_action[:, 9:] 
            # #print(next_state.shape)
            for i in range(16):
                action = get_action(poses[i], gripper[i])
                print(action)
                action_list.append(action)
                #action = self.actions[i*16+j]
                obs, _, _, _,info = self.env.step(action)
                self.env.render()
                self.buffer.update(self.get_current_obs())
            frame, _, _ = get_relevant_info(self.env,  self.initial_pose)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)
        return None

    
from torchvision import transforms

def load_model(checkpoint_path: str, device: str):
    # build model dict
    resnet = replace_bn_with_gn(get_resnet('resnet18'))
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = nn.ModuleDict({
            'vision_encoder': resnet,
            'lldm': create_model(vocab_size=99, d_model=768, n_heads=12, n_layers=12,
                 cond_dim=2*512*49+2*10, num_latents=64).to(device)
        })

    if os.path.isfile(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='raw_data/bottle_rows/2025_03_04_13_51_38_PickAnything.json')
    parser.add_argument('--checkpoint', type=str, default='modelbias_epoch50.pt')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=10)
    args = parser.parse_args()

    # load JSON and create env with rgb_array output
    json_data = io_utils.load_json(args.json_path)
    json_data['env_info']["env_kwargs"]['show_goal_site']= False
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    #ori_env_kwargs['render_mode'] = 'rgb_array'
    ori_env = gym.make(env_id, **ori_env_kwargs)
    ori_env.reset(**json_data['episodes'][args.episode]['reset_kwargs'])

    # initialize video writer
    initial_pose = np.array([[1,0,0,0.097], [0,1,0,0], [-0.1,0,1,0.378], [0,0,0,1]], dtype=np.float32)
    first_frame, _, _ = get_relevant_info(ori_env= ori_env, initial_pose=initial_pose)
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('output.mp4', fourcc, 30, (w, h))
    # write the first frame
    writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # load models and runner
    
    tokenizer = TemporalBPEProcessor.load("saved_processor")
    model = load_model(args.checkpoint, device)
    codec = StateCodec(tokenizer, max_length=512)
    actions = np.load('actions.npy')

    runner = SimModelRunner(actions, ori_env, model, tokenizer, codec, initial_pose, writer, device)

    # run loop and record
    for step in range(args.max_steps):
        runner.step(j=step)
    
    all_actions = np.stack(action_list, axis=0)   # shape: [T, action_dim]
    np.save("actions.npy", all_actions)

    writer.release()
    ori_env.close()

if __name__ == '__main__':
    main()
