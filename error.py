import copy
import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fail_observer import FailObserver
from model import *
from model.constants import *
from model.dataset import (OriginalMAPS, SynthesizedInstruments,
                           SynthesizedTrumpet, CustomBatchDataset)
from model.evaluate_fn import evaluate_wo_velocity
from snapshot import Snapshot

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(34)

train_dataset = SynthesizedInstruments(dataset_root_dir=".", groups=["train"],
    sequence_length=327680, device="cuda", refresh=False)
validation_dataset = SynthesizedInstruments(dataset_root_dir=".", groups=["val"],
    sequence_length=327680, device="cuda", refresh=False)
test_dataset = SynthesizedInstruments(dataset_root_dir=".", groups=["test"],
    sequence_length=327680, device="cuda", refresh=False)

loader = DataLoader(train_dataset, 32, shuffle=True, drop_last=True)
valloader = DataLoader(validation_dataset, 4, shuffle=False, drop_last=True)

# batch_visualize = next(iter(valloader))

model = UnetTranscriptionModel(
    ds_ksize=(2, 2), ds_stride=(2, 2), mode="imagewise",
    spec='CQT', norm=1, device="cuda", logdir="unet_temp", debug_mode=False
)

model.load_my_state_dict(torch.load("results/unet_model_trained_on_SynthesizedInstruments/SNAPSHOT/372_model.pt"))
model.to("cuda")

new_eps = 1e-4
for m in model.modules():
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        print(f"Adjusting eps of {m} from {m.eps} to {new_eps}")
        m.eps = new_eps

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.load_state_dict(torch.load("results/unet_model_trained_on_SynthesizedInstruments/SNAPSHOT/372_optimizer.pt"))

scheduler = StepLR(optimizer, step_size=10000, gamma=0.98)
scheduler.load_state_dict(torch.load("results/unet_model_trained_on_SynthesizedInstruments/SNAPSHOT/372_scheduler.pt"))

batch_size = 32
total_batch = len(loader.dataset)

ep = 373

model.train()
total_loss = 0
batch_idx = 0

for batch in loader:
    optimizer.zero_grad()

    predictions, losses, _ = model.run_on_batch(batch, batch_description=str(ep), batch_identifier=batch_idx)

    loss = sum(losses.values())
    total_loss += loss.item()
    loss.backward()

    nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
    optimizer.step()
    scheduler.step()

    batch_idx += 1
    print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
            f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
            f'\tLoss: {loss.item():.6f}')

# # epoch_loss = total_loss/len(loader)
# # print(f'Train Epoch: {ep}\tLoss: {epoch_loss:.6f}')

# # predictions, losses, mel = model.run_on_batch(batch_visualize, "evaluation")