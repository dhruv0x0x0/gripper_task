import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils import replace_bn_with_gn, mat_to_pose9d
import os

class LowdimMaskGenerator(nn.Module):
    def __init__(self,
        action_dim, obs_dim,
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        action_visible=False
        ):

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__() 

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask


from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
logger = logging.getLogger(__name__)
from utils import get_resnet
from u_net import ConditionalUnet1D

from normalizer import DataNormalizer
data_dir = 'out_dataset_bottle'
save_dir = 'saved_normaliser'
normalizer = DataNormalizer.fit_from_npz(data_dir, sample_limit=-1)
normalizer.save(save_dir)

class DiffusionPolicy():
    def __init__(self,  
            noise_scheduler: DDPMScheduler,
            horizon = 16, 
            n_action_steps = 14, 
            n_obs = 2,
            num_inference_steps=None,
            lowdim_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            load_path = None):
       
        action_shape = (10, 16)
        action_dim = action_shape[0]
        # obs_shape_meta = shape_meta['obs']
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rgb_net =replace_bn_with_gn(get_resnet('resnet18'))
        rgb_feature_dims = 512
        lowdim_input_dim = 10
        self.lowdim_net = None

        # compute dimensions for diffusion
        rgb_feature_dim = rgb_feature_dims * n_obs
        lowdim_input_dim = lowdim_input_dim * n_obs
        global_cond_dim = rgb_feature_dim + lowdim_input_dim
        input_dim = action_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
    #     tcn = nn.Sequential(
    # nn.Conv1d(in_channels=10, out_channels=128, kernel_size=2),
    # nn.ReLU(),
    # nn.AdaptiveAvgPool1d(1),
    # nn.Flatten(), 
    # nn.Linear(128,512)
#)
        self.model = nn.ModuleDict({
            'vision_encoder': self.rgb_net,
            'unet': model,
            
        })

        if load_path and os.path.isfile(load_path): 
            print(f"Loading model from {load_path}")
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = DataNormalizer.load(save_dir)
        self.horizon = horizon
        self.action_dim = action_dim
        self.lowdim_input_dim = lowdim_input_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs
        self.lowdim_as_global_cond = lowdim_as_global_cond
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        self.model.to(self.device)
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model['unet']
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs):
        gripper_state = torch.cat((obs['pose'], obs['gripper']), dim=-1)
        low_dim = self.normalizer.normalize(gripper_state)
        lowdim_feature = low_dim.reshape(low_dim.shape[0], -1).to(self.device)#.permute(0,2,1).to(self.device)#.reshape(low_dim.shape[0],-1)
        rgb_img = torch.tensor(obs['image']).to(self.device).view(-1, 3, 224, 224)
        rgb_feature = self.model['vision_encoder'](rgb_img)
        rgb_feature = torch.cat(torch.chunk(rgb_feature, 2, dim=0) ,dim=-1)
        #lowdim_feature = self.model['temp'](low_dim)

        B, To, _ = gripper_state.shape
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device

        global_cond = None
        cond_data = None
        cond_mask = None
   
        global_cond = torch.cat([rgb_feature, lowdim_feature], dim=-1)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=torch.float32)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
      
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=None,
            global_cond=global_cond)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer.unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    def get_obs_features(self, batch):
        rgb_img = torch.tensor(batch['obs']['image']).to(self.device)
        gripper_state = torch.cat((batch['obs']['pose'], batch['obs']['gripper']), dim=-1)
        qq,yy,ss = gripper_state.shape
        gripper_state = gripper_state.reshape(-1, yy*ss)
        gripper_state = torch.tensor(gripper_state).to(self.device)
        rgb_img = rgb_img.view(-1, 3, 224, 224)
        image_features = self.model['vision_encoder'](rgb_img)
        image_features = torch.cat(torch.chunk(image_features, 2, dim=0) ,dim=-1)
        obs_features = torch.cat([image_features, gripper_state], dim=-1)
        return obs_features
    
    def compute_loss(self, batch):
        gripper_state = torch.cat((batch['obs']['pose'], batch['obs']['gripper']), dim=-1)
        low_dim = self.normalizer.normalize(gripper_state)
        #low_dim = low_dim.permute(0,2,1).to(self.device)#.reshape(low_dim.shape[0],-1)
        lowdim_feature = low_dim.reshape(low_dim.shape[0], -1).to(self.device)

        nactions = self.normalizer.normalize(batch['target_state']).to(self.device)
        rgb_img = torch.tensor(batch['obs']['image']).to(self.device).view(-1, 3, 224, 224)
        rgb_feature = self.model['vision_encoder'](rgb_img)
        rgb_feature = torch.cat(torch.chunk(rgb_feature, 2, dim=0) ,dim=-1)
        #lowdim_feature = self.model['temp'](low_dim)
        #print(lowdim_input.shape)
        # handle different ways of passing lowdim
        global_cond = None
        trajectory = None
        cond_data = None
        global_cond = torch.cat([rgb_feature, lowdim_feature], dim=-1)
        trajectory = nactions
        cond_data = nactions
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model['unet'](noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond)

        #if self.kwargs.get('predict_epsilon', True):
            # default for most methods
        target = noise
        # else:
        #     # DDPM also has
        #     target = trajectory

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss





