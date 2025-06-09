from torchvision import transforms
import torch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
from model import create_model
from data import ManiSkillSequenceDataset
from tokeniser import TemporalBPEProcessor
from vision_model_getter import get_resnet
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from tokeniser import TemporalBPEProcessor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from model import create_model
from lldm_train import replace_bn_with_gn, generate
from vision_model_getter import get_resnet
tokeniser = TemporalBPEProcessor.load("saved_tokenizer_pickanything")

import os
model = nn.ModuleDict({
    'vision_encoder': replace_bn_with_gn(get_resnet('resnet18')),
    'lldm': create_model(vocab_size=1500, d_model=768, n_heads=12, n_layers=12)
})
load_path = 'model_epoch1.pt'
if load_path and os.path.isfile(load_path):  # <-- Load model if specified
    print(f"Loading model from {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
out = generate(model, input_ids, steps=128, gen_length=128, block_length=32,
                temperature=0.0, cfg_scale=0.0)
print(tokeniser.batch_decode(out[:, input_ids.size(1):], skip_special_tokens=True)[0]) 
        # # 3) diffusion + conditioning
        # input_ids = None#self.past_tokens.unsqueeze(0)            # [1, seq_len]
        
