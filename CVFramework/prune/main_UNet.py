import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from models.simpleUnet import UNet

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from models.unet_advanced import pixel_accuracy, MiOU, fit
from data.drone import DroneDataset, DroneTestDataset, IMAGE_PATH, MASK_PATH, mean, std

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda")


def create_df(path):
    name = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df(IMAGE_PATH)


X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=0)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=0)

print('Train Size   : ', len(X_train)) #306
print('Val Size     : ', len(X_val)) #54
print('Test Size    : ', len(X_test)) #40

#create datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)
#load data
batch_size= 3

#create dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

#create model
model = UNet()


lr = 0.001
epoch = 30
weight_decay = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy = fit(
    epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, batch_size
)

train_log = {'train_loss' : train_losses, 'val_loss': val_losses,
        'train_miou' :train_miou, 'val_miou':val_miou,
        'train_acc' :train_accuracy, 'val_acc':val_accuracy}

# save the model
torch.save(model, 'Unet_false.pt', _use_new_zipfile_serialization=False)
torch.save(model, 'Unet_true.pt', _use_new_zipfile_serialization=True)
