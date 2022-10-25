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


device = torch.device("cuda")




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



#############################################################################################
state_dict_path = "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/UNet_26.pth"
print("We are using the state dict from\n", state_dict_path)
#############################################################################################




criterion = nn.CrossEntropyLoss()
print("Prune the ENCODER ")
print("Encoder pruning\n ratio: 0 - 1 \nmethod: Global Unstructured L1")




outcome = {}
outcome_pruned = {}
outcome_retrain = {}

for ratio in [0.4, 0.5, 0.6]:
    model = UNet(out_classes=24).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))


    # test the behavior of model before pruning 
    total_loss = 0
    total_acc = 0
    total_m = 0

    start = time.time()

    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        acc_i = pixel_accuracy(pred, mask)
        m_i = MiOU(pred, mask, n_classes=24)
        total_loss += loss.item()
        total_m += m_i
        total_acc += float(acc_i)
    
    end = time.time()

    duration = end - start

    loss, acc, m = total_loss/len(val_loader), total_acc/len(val_loader), total_m/ len(val_loader)
    
    print('before pruning and tuning:')
    print("loss: ", loss)
    print("acc: ", acc)
    print("MIOU: ", m)
    print("Duration: ", duration)

    outcome[ratio] = (loss, acc, m, duration)



    # begin pruning
    print('sparsity after pruning: ')
    parameters_to_prune_encoder = (
    # (model.double_conv1[0], 'weight'),
    # (model.double_conv1[2], 'weight'),

    # (model.double_conv2[0], 'weight'),
    # (model.double_conv2[2], 'weight'),

    (model.double_conv3[0], 'weight'),
    (model.double_conv3[2], 'weight'),

    (model.double_conv4[0], 'weight'),
    (model.double_conv4[2], 'weight'),
    )
    
    prune.global_unstructured(
    parameters_to_prune_encoder,
    pruning_method=prune.RandomUnstructured,
    amount=ratio,
    )
    for module, dummy in parameters_to_prune_encoder:
        # prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )
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
    prune.global_unstructured(
    parameters_to_prune_decoder,
    pruning_method=prune.RandomUnstructured,
    amount=ratio,
    )
    print("AMOUNT: ", ratio)

    for module, dummy in parameters_to_prune_decoder:
        # prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )

    # test the behavior of model after pruning 
    total_loss = 0
    total_acc = 0
    total_m = 0

    start = time.time()

    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        acc_i = pixel_accuracy(pred, mask)
        m_i = MiOU(pred, mask, n_classes=24)
        total_loss += loss.item()
        total_m += m_i
        total_acc += float(acc_i)
    
    end = time.time()

    duration = end - start

    loss, acc, m = total_loss/len(val_loader), total_acc/len(val_loader), total_m/ len(val_loader)
    
    print('after pruning:')
    print("loss: ", loss)
    print("acc: ", acc)
    print("MIOU: ", m)
    print("Duration: ", duration)

    outcome_pruned[ratio] = (loss, acc, m, duration)



    # tuning after pruning
    start = time.time()
    
    print("Re-train the pruned model{}".format(ratio))

    lr = 0.001
    epochs = 30
    weight_decay = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    fit(
        epochs,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        batch_size,
        n_class=24
    )
    # sparsity ater tuning
    print('sparsity after tuning: ')
    for module, dummy in parameters_to_prune_encoder:
        # prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )

    for module, dummy in parameters_to_prune_decoder:
        # prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )

    total_loss = 0
    total_acc = 0
    total_m = 0

    # accuracy after tuning
    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        acc_i = pixel_accuracy(pred, mask)
        m_i = MiOU(pred, mask, n_classes=24)
        total_loss += loss.item()
        total_m += m_i
        total_acc += float(acc_i)
    
    end = time.time()

    duration = end - start

    loss, acc, m = total_loss/len(val_loader), total_acc/len(val_loader), total_m/ len(val_loader)
    
    print('after tuning:')
    print("loss: ", loss)
    print("acc: ", acc)
    print("MIOU: ", m)
    print("Duration: ", duration)

    outcome_retrain[ratio] = (loss, acc, m, duration)
    print("Done for fitting")


   
print(outcome)
print(outcome_pruned)
print(outcome_retrain)



