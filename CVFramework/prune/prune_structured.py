#%%
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

print("all done")

#%%
# from thop import profile

device = torch.device("cuda")
# device = torch.device("cuda")


IMAGE_PATH = '/scratch/qz1086/drone_dataset/new_size_img/'
MASK_PATH = '/scratch/qz1086/drone_dataset/new_size_mask/'

save_dir = "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/12_14/"


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

# model = UNet(out_classes=24).to(device)


#############################################################################################
state_dict_path = "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/UNet_26.pth"
print("We are using the state dict from\n", state_dict_path)
#############################################################################################



# experiment 1, 2, 3, 4======================================================================
criterion = nn.CrossEntropyLoss()
decoder_ratio_bin = np.linspace(0.0, 1.0, 9)
# prune_method_bin = [prune.L1Unstructured, prune.RandomUnstructured]
prune_method_bin = [1, 2]   # ['random']

print('this is the trial for L1, L2 structred pruning, \t\
        prune encoder in np.linspace(0.0, 1.0, 9), \t\
        decoder in np.linspace(0.0, 1.0, 9)')
print('the matrix notation: first for L1, second for L2, \t\
        raw for decoder, colum for encoder')
trial_out = np.zeros(2*9*9).reshape(2, 9, 9)
for a in range(len(prune_method_bin)):
    prune_method = prune_method_bin[a]
    print("Encoder pruning\n amount: 0.0 - 1.0 \nmethod: {}".format(prune_method))

    
    for b in range(len(decoder_ratio_bin)):
        decoder_ratio = decoder_ratio_bin[b]
        print("decoder pruning ratio = {:.2f}".format(decoder_ratio))

        
        for c in range(9):
            amount = np.linspace(0.0, 1.0, 9)[c]
            print('encoder pruning ratio = {}'.format(amount))
            model = UNet(out_classes=24).to(device)
            model.load_state_dict(torch.load(state_dict_path, map_location=device))
            
            # prune different rate for encoder
            parameters_to_prune_encoder = (
                    (model.double_conv1[0], 'weight'),
                    (model.double_conv1[2], 'weight'),

                    (model.double_conv2[0], 'weight'),
                    (model.double_conv2[2], 'weight'),

                    (model.double_conv3[0], 'weight'),
                    (model.double_conv3[2], 'weight'),

                    (model.double_conv4[0], 'weight'),
                    (model.double_conv4[2], 'weight'),
                    )
            
            # prune.global_unstructured(
            # parameters_to_prune_encoder,
            # pruning_method=prune_method, # L1 or L2
            # amount=amount,
            # )
            # print('the following is the encoder module sparsity info: ')
            for module, dummy in parameters_to_prune_encoder:

                module = prune.ln_structured(
                    module, 'weight', amount=amount, dim=0, n=prune_method)
                prune.remove(module, 'weight')
                # print(
                # "Density in " + str(module)+ " : {:.2f}%".format(
                #     100. * float(torch.sum(module.weight != 0))
                #     / float(module.weight.nelement())
                # )
                # )

            # prune 0.2 or 0.25 for decoder
            parameters_to_prune_decoder = (
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

            # for decoder, we use unstructred L1/L2
            # prune.global_unstructured(
            # parameters_to_prune_decoder,
            # pruning_method=prune_method,
            # amount=decoder_ratio,
            # )
            # print('the following is the encoder module sparsity info: ')
            for module, dummy in parameters_to_prune_decoder:

                module = prune.ln_structured(
                    module, 'weight', amount=decoder_ratio, dim=0, n=prune_method)
                    
                prune.remove(module, 'weight')
                # print(
                # "Density in " + str(module)+ " : {:.2f}%".format(
                #     100. * float(torch.sum(module.weight != 0))
                #     / float(module.weight.nelement())
                # )
                # )

        
            # test the loss after pruning for encoder and decoder
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

            trial_out[a][b][c] = (
                total_val_loss/len(val_loader)
            )

print(trial_out[0])
print(trial_out[1])


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



# print("Prune the DECODER only...")
# model.load_state_dict(torch.load(state_dict_path, map_location=device))



# decoder_out = {}
# for amount in np.linspace(0, 1, 9):
#     model = UNet(out_classes=24).to(device)
#     model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
#     parameters_to_prune = (
#             (model.up_conv1, 'weight'),
#             (model.up_conv2, 'weight'),
#             (model.up_conv3, 'weight'),
#             (model.up_conv4, 'weight'),

#             (model.up_double_conv1[0], "weight"),
#             (model.up_double_conv1[2], "weight"),

#             (model.up_double_conv2[0], "weight"),
#             (model.up_double_conv2[2], "weight"),

#             (model.up_double_conv3[0], "weight"),
#             (model.up_double_conv3[2], "weight"),

#             (model.up_double_conv4[0], "weight"),
#             (model.up_double_conv4[2], "weight"),
#             )

#     prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=amount,
#     )
#     for module, dummy in parameters_to_prune:
#         prune.remove(module, 'weight')
#         print(
#         "Density in " + str(module)+ " : {:.2f}%".format(
#             100. * float(torch.sum(module.weight != 0))
#             / float(module.weight.nelement())
#         )
#         )


#     total_val_loss = 0
#     total_val_miou = 0
#     total_val_accuracy = 0
#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):
#             img, mask = batch
#             # !!tensor and model are different, not inplace 
#             img = img.to(device)
#             mask = mask.to(device)

#             prediction = model(img)
#             loss = criterion(prediction, mask)  # how to split mask from the train_loader
#             total_val_loss += loss.item()

#             total_val_miou += MiOU(prediction,mask,24)
#             total_val_accuracy += pixel_accuracy(prediction, mask)

#             total_val_loss += loss.item()
                    
#         print(total_val_loss/len(val_loader))
#         print('sparcity of model: ')
        
#         module = model.up_conv1
#         print(
#         "Density in " + str(module)+ " : {:.2f}%".format(
#             100. * float(torch.sum(module.weight != 0))
#             / float(module.weight.nelement())
#         ))

#         decoder_out[amount] = (
#             total_val_loss/len(val_loader)
#         )

# for i in decoder_out.items():
#     print(i)


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

