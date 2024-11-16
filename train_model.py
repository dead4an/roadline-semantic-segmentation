import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from safetensors.torch import save_file
from tqdm import tqdm
from utils import RoadlineTrainDataset


# Paths
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'img')
ANN_DIR = os.path.join(DATA_DIR, 'ann')
CHECKPOINT_PATH =  os.path.join(ROOT, 'checkpoints', 'unet_segment.safetensors')

# Torch settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Train settings
NUM_EPOCHS = 30
BATCH_SIZE = 8
LR = 0.02
LR_STEP_SIZE = 10
LR_STEP = 0.1

# Data
train_dataset = RoadlineTrainDataset(
    img_dir=IMG_DIR,
    ann_dir=ANN_DIR,
    orig_shape=(1920, 1080), # orig shape is (width, height)
    resize_shape=(736, 416), # resize shape is (width, height)
    device=DEVICE
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
).to(DEVICE)


# Optimization
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=LR_STEP_SIZE,
    gamma=LR_STEP
)

criterion = nn.BCELoss()

# Train
for epoch in range(1, NUM_EPOCHS + 1):
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
    total_loss = 0

    for idx, batch in enumerate(progress_bar, start=1):
        img, mask = batch
        output = model(img).squeeze(1)
        output = F.sigmoid(output)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss
        progress_bar.set_postfix({'loss': total_loss.item() / idx})

    scheduler.step()

model_state_dict = model.state_dict()
save_file(model_state_dict, CHECKPOINT_PATH)
