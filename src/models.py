import torch

import zipfile
from pathlib import Path

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import datetime
import random
import cv2

import torch
import torchvision.models 
import torchvision.transforms as transforms
from PIL import Image



def get_pretrained_VGG16():
    # Загружаем предварительно обученную модель VGG16
    vgg16 = torchvision.models.vgg16(pretrained=True)
    # нужное число выходных нейронов
    OUTPUT_NEURONS = 5

    # У вгг последние слои находятся в блоке classifier и он состоит из 6 слоев, мы обращаемся
    # к этому блоку, к последнему (6) слою и забираем in_features, так как он понадобится для изменения 
    # последнего слоя
    num_of_in_features = vgg16.classifier[6].in_features

    # заменяем число выходных нейронов своим числом
    vgg16.classifier[6] = torch.nn.Linear(num_of_in_features, OUTPUT_NEURONS)

    return vgg16

def get_efficient_net(size = 'm', pretrained=True):
    print('Входное должно иметь размер 224x224')

    models = {'s':torchvision.models.efficientnet_v2_s(pretrained=pretrained),
             'm': torchvision.models.efficientnet_v2_m(pretrained=pretrained),
             'l':torchvision.models.efficientnet_v2_l(pretrained=pretrained)}
    
    effnet = models[size]

    OUTPUT_NEURONS = 5

    num_of_in_features = effnet.classifier[-1].in_features

    effnet.classifier[-1] = torch.nn.Linear(num_of_in_features, OUTPUT_NEURONS)

    return effnet
    

def get_resnet(pretrained):

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)

    OUTPUT_NEURONS = 5

    num_of_in_features = resnet.fc.in_features


    resnet.fc = torch.nn.Linear(num_of_in_features, OUTPUT_NEURONS)

    return resnet
    

class VGG16(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.img_size = img_size

        self.conv_layers = nn.ModuleList([])

        layers_with_maxpool = [2, 4, 7, 10, 13]
        kernel_count = {1: 64, 2: 64, 3: 128, 4: 128, 5: 256, 6: 256,
                        7: 256, 8: 512, 9: 512, 10: 512, 11: 512, 12: 512, 13: 512}

        for layer in range(1, 14):
            cur_kernel_count = kernel_count[layer]
            self.conv_layers.append(nn.Conv2d(cur_kernel_count, cur_kernel_count, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(cur_kernel_count))
            self.conv_layers.append(nn.ReLU(inplace=True))

            if layer in layers_with_maxpool:
                self.conv_layers.append(nn.MaxPool2d(2, 2))

        self.fc_layers = nn.ModuleList([])
        neuron_count = {14: (self.img_size/2**5)*512, 15: 4096}

        for layer in range(14, 16):
            cur_neuron_count = neuron_count[layer]
            self.fc_layers.append(nn.Dropout(0.5))
            self.fc_layers.append(nn.Linear(int(cur_neuron_count), 4096))
            self.fc_layers.append(nn.ReLU(inplace=True))

        self.fc_layers.append(nn.Linear(4096, 5))  # 5 - P, x, y, w, h

    def forward(self, out):

        for cur_layer in self.conv_layers:
            out = cur_layer(out)

        out = torch.flatten(out, 1)

        for cur_layer in self.fc_layers:
            out = cur_layer(out)

        return out
