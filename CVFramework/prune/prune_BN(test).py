from math import e
import os
from re import I
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

import torch.nn.utils.prune as prune

from torch.autograd import Variable
import util

# import tensorflow as tf
# import segmentation_models_pytorch as smp

# for image augmentation
# import albumentations as A

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from models.simpleUnet import UNet
from data.drone import DroneDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from models.unet_advanced import pixel_accuracy, MiOU, fit

# from thop import profile

device = torch.device("cuda")
# device = torch.device("cuda")


IMAGE_PATH = '/scratch/qz1086/drone_dataset/new_size_img/'
MASK_PATH = '/scratch/qz1086/drone_dataset/new_size_mask/'

save_dir = "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/12_12/"


# create df with id of the dataset
def create_df(path):
    name = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df(IMAGE_PATH)
# print('Total Images: ', len(df))

#split the dataset into train, validation and test data
X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=0)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=0)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


#create datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)
#load data
batch_size= 3

# train_loader = DataLoader(train_set)
# val_loader = DataLoader(val_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


#############################################################################################
state_dict_path = "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/UNet_26.pth"
print("We are using the state dict from\n", state_dict_path)

model_origin = UNet(out_classes=24).to(device)
print('original model structure:\n', model_origin)
#############################################################################################

total = 0
i = 0
for m in model.modules():
        # 如果是BN层统计一下通道
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                total += m.weight.data.shape[0]


criterion = nn.CrossEntropyLoss()
print("Prune the ENCODER only...")
print("Encoder pruning\n amount: 0 - 1 \nmethod: Global Unstructured L1")



encoder_out = {}
for amount in np.linspace(0, 1, 9):
    model = UNet(out_classes=24).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
    parameters_to_prune = (
            (model.double_conv1[0], 'weight'),
            (model.double_conv1[2], 'weight'),

            (model.double_conv2[0], 'weight'),
            (model.double_conv2[2], 'weight'),

            (model.double_conv3[0], 'weight'),
            (model.double_conv3[2], 'weight'),

            (model.double_conv4[0], 'weight'),
            (model.double_conv4[2], 'weight'),
            )
            
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=amount,
    )


    for module, dummy in parameters_to_prune:
        prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )

        

    total_val_loss = 0
    total_val_miou = 0
    total_val_accuracy = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img, mask = batch
            # !!tensor and model are different, not inplace 
            img = img.to(device)
            mask = mask.to(device)

            prediction = model(img)
            loss = criterion(prediction, mask)  # how to split mask from the train_loader
            total_val_loss += loss.item()

            total_val_miou += MiOU(prediction,mask,24)
            total_val_accuracy += pixel_accuracy(prediction, mask)

            total_val_loss += loss.item()
                    
 

        print(total_val_loss/len(val_loader))

        encoder_out[amount] = (
            total_val_loss/len(val_loader)
        )

for i in encoder_out.items():
    print(i)

# print("Re-train the pruned model(endcoder pruned)")

# lr = 0.001
# epochs = 30
# weight_decay = 0.0001
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs,
#                                             steps_per_epoch=len(train_loader))


# fit(
#     epochs,
#     model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     scheduler,
#     batch_size,
#     n_class=24,
#     device=device,
#     save_dir=save_dir,
#     file_name="Unet_50_EP" # which means, Unet, 50% sparsity, Encoder pruned.
#     ) 



# print("Done for pruning only the ENCODER")

# print("Now, let's huohuo DECODER")



print("Prune the DECODER only...")
model.load_state_dict(torch.load(state_dict_path, map_location=device))



decoder_out = {}
for amount in np.linspace(0, 1, 9):
    model = UNet(out_classes=24).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
    parameters_to_prune = (
            (model.up_conv1, 'weight'),
            (model.up_conv2, 'weight'),
            (model.up_conv3, 'weight'),
            (model.up_conv4, 'weight'),

            (model.up_double_conv1[0], "weight"),
            (model.up_double_conv1[2], "weight"),

            (model.up_double_conv2[0], "weight"),
            (model.up_double_conv2[2], "weight"),

            (model.up_double_conv3[0], "weight"),
            (model.up_double_conv3[2], "weight"),

            (model.up_double_conv4[0], "weight"),
            (model.up_double_conv4[2], "weight"),
            )

    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=amount,
    )
    for module, dummy in parameters_to_prune:
        prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )


    total_val_loss = 0
    total_val_miou = 0
    total_val_accuracy = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img, mask = batch
            # !!tensor and model are different, not inplace 
            img = img.to(device)
            mask = mask.to(device)

            prediction = model(img)
            loss = criterion(prediction, mask)  # how to split mask from the train_loader
            total_val_loss += loss.item()

            total_val_miou += MiOU(prediction,mask,24)
            total_val_accuracy += pixel_accuracy(prediction, mask)

            total_val_loss += loss.item()
                    
        print(total_val_loss/len(val_loader))
        print('sparcity of model: ')
        
        module = model.up_conv1
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        ))

        decoder_out[amount] = (
            total_val_loss/len(val_loader)
        )

for i in decoder_out.items():
    print(i)


# print("Re-train the pruned model(decoder pruned)")

# lr = 0.001
# epochs = 30
# weight_decay = 0.0001
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs,
#                                             steps_per_epoch=len(train_loader))

# fit(
#     epochs,
#     model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     scheduler,
#     batch_size,
#     n_class=24,
#     device=device,
#     save_dir=save_dir,
#     file_name="Unet_50_DP" # which means, Unet, 50% sparsity, Decoder pruned.
# )


# print("Done for DECODER")

