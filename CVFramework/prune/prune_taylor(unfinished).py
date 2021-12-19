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
from DataUtils import FilterPrunner

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from models.unet_advanced import pixel_accuracy, MiOU, fit

# from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
X_trainval, X_test = train_test_split(df['id'].values, test_size=0.05, random_state=0)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=0)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


#create datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)
test_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std)
#load data
batch_size= 3

# train_loader = DataLoader(train_set)
# val_loader = DataLoader(val_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# model = UNet(out_classes=24).to(device)


#=============================================================================================
state_dict_path = "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/UNet_26.pth"
print("We are using the state dict from\n", state_dict_path)

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def reset(self):
        self.filter_ranks = {}
            

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.named_modules()):
            x = module(x)
            if isinstance(module, torch.nn.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return x
   
    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (image, mask) in enumerate(self.train_data_loader):
            img = img.to(device)
            mask = mask.to(device)

            prediction = model(img)  # pass batch
            loss = criterion(prediction, mask)  # calculate loss, loss tensor
            self.model.train()

            if rank_filters:
                output = self.forward(img)
                self.criterion(img, mask).backward()
            else:
                self.criterion(self.model(img), mask).backward()


    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()
]
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune      
    
    
    def get_candidates_to_prune(self, num_filters_to_prune):
        self.reset()
        self.train_epoch(rank_filters = True)
        self.normalize_ranks_per_layer()
        return self.get_prunning_plan(num_filters_to_prune)       


    def prune(self):
        #Get the accuracy before prunning
        # test on validation set

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512

        print("Number of prunning iterations to reduce 67 /%/ filters", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)
        torch.save(model.state_dict(), "model_prunned")



#=============================================================================================





print('the original model structure: ')
model.summary()


criterion = nn.CrossEntropyLoss()
print("Prune the ENCODER only...")
print("Encoder pruning\n amount: 0 - 1 \nmethod: Global Unstructured L1")


encoder_out = {}

# test different ratio of pruning
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

    # prune.global_unstructured(
    # parameters_to_prune,
    # pruning_method=prune.L1Unstructured,
    # amount=amount,
    # )

    # pruning method: Taylor sorting method
    for module, _ in parameters_to_prune:
        x = module(x)
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            x.register_hook(self.compute_rank)
            self.
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


print(encoder_out)

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

decoder_out = {}
for amount in np.linspace(0, 1, 9):
    model = UNet(out_classes=24).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
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

        decoder_out[amount] = (
            total_val_loss/len(val_loader)
        )

print(decoder_out)


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

