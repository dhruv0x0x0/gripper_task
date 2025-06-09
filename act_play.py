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
def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

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
    def __init__(self,
                 env: gym.Env,
                 model: nn.ModuleDict,
                 tokenizer: TemporalBPEProcessor,
                 state_codec: StateCodec,
                 initial_pose: np.ndarray,
                 device: str = 'cpu'):
        self.env = env
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.codec = state_codec
        self.initial_pose = initial_pose
        self.device = device

        # Preprocessing for ResNet
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485,0.456,0.406],
            #                      std=[0.229,0.224,0.225]),
        ])

    # def reset(self, **reset_kwargs):
    #     obs_state, info = self.env.reset(**reset_kwargs)
    #     # encode initial simulator state to tokens
    #     # init_state_dict = self.env.get_state_dict()
    #     # self.past_tokens = self.codec.state_to_tokens(init_state_dict)
    #     return obs_state, info

    @torch.no_grad()
    def step(self):
        # 1) get vision + proprioceptive observation
        img_np, relT, grip = get_relevant_info(self.env, self.initial_pose)
        img = self.preprocess(img_np).unsqueeze(0).to(self.device)  # [1,3,224,224]
        proprio = torch.from_numpy(
            np.concatenate([mat_to_pose10d(relT), np.array([grip])])
        ).unsqueeze(0).to(self.device)  # [1, 16+1]
     
        # 2) forward through vision encoder
        vis_feats = self.model['vision_encoder'](img) 
        B, C, H, W = vis_feats.shape
        vis_feats = vis_feats.reshape(B, C*H*W)       # e.g. [1, feat_dim]
        obs_feats = torch.cat([vis_feats, proprio], dim=-1)  # [1, feat_dim + 17]

        # 3) diffusion + conditioning
        input_ids = None#self.past_tokens.unsqueeze(0)            # [1, seq_len]
        

        new_tokens = generate(model =self.model['lldm'], prompt= input_ids, cond= obs_feats, mask_id = 0, device= device)#logits = self.model['lldm'](input_ids, obs_feats)      # [1, seq_len, vocab]

        # 4) pick next token(s)
        # next_token = logits.argmax(dim=-1)[:, -1:]           # [1,1]
        # new_tokens = next_token.squeeze(0).tolist()          # List[int]

        # 5) decode tokens → next state dict
        next_action= self.codec.tokens_to_state(new_tokens).squeeze(0)
        poses =  next_action[:, :9]   # first 9 columns
        poses =  pose10d_to_mat(poses)
        gripper = next_action[:, 9:] 
        #print(next_state.shape)
        for i in range(16):
           action = get_action(poses[i], gripper[i])
           action_list.append(action)
           obs, _, _, _,info = self.env.step(action)
           self.env.render()
           #state = self.env.get_state_dict()
           #print(next_state.shape)
           #print(state)
           #state['articulations']['floating_gripper_v1'] = next_state[0][i]
           #ns = {'actors':{},'articulations':{'floating_gripper_v1':next_state[0][i]}}

        #    self.env.set_state_dict(state)
        #    self.env.render()
           time.sleep(0.1)
        # self.past_tokens = torch.cat([
        #     self.past_tokens, next_token.squeeze(0)
        # ], dim=0)
        # # optional: keep buffer length <= max_length
        # if self.past_tokens.size(0) > self.codec.max_length:
        #     self.past_tokens = self.past_tokens[-self.codec.max_length:]

        return next_action
from torchvision import transforms

def load_model(checkpoint_path: str, device: str):
    # build model dict
    resnet = replace_bn_with_gn(get_resnet('resnet18'))
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = nn.ModuleDict({
            'vision_encoder': resnet,
            'lldm': create_model(vocab_size=500, d_model=768, n_heads=12, n_layers=12,
                 cond_dim=512*49+10, num_latents=64).to(device)
        })

    if os.path.isfile(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='raw_data/bottle_rows/2025_03_04_13_51_38_PickAnything.json',
                        help='Original traj JSON with env_info & reset_kwargs')
    parser.add_argument('--checkpoint', type=str,
                        default='model_epoch2.pt',
                        help='Model checkpoint to load')
    parser.add_argument('--episode', type=int, default=0,
                        help='Which episode idx from JSON to run')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum model-env loop steps')
    args = parser.parse_args()

    # 1) load JSON meta
    json_data = io_utils.load_json(args.json_path)
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_info['env_kwargs']['render_mode']= "rgb_array"
    ori_env_kwargs = env_info["env_kwargs"]
    ori_env = gym.make(env_id, **ori_env_kwargs)
    episode = json_data["episodes"][0] 
    ori_env.reset(**episode["reset_kwargs"])
    initial_pose = np.array([
    [1,0,0,0.097],
    [0,1,0,0],
    [-0.1,0,1,0.378],
    [0,0,0,1]
], dtype=np.float32)
   
    tokenizer = TemporalBPEProcessor.load("saved_processor")
    model = load_model(args.checkpoint, device)

    # 4) build codec & runner
    codec = StateCodec(tokenizer, max_length=512)
    runner = SimModelRunner(ori_env, model, tokenizer, codec, initial_pose, device)

    # 5) reset
    # 6) run loop
    for step in range(args.max_steps):
        next_state = runner.step()
        # insert any termination logic here, e.g. break on success/failure

    all_actions = np.stack(action_list, axis=0)   # shape: [T, action_dim]
    np.save("actions.npy", all_actions)
    ori_env.close()


if __name__ == '__main__':
    main()
