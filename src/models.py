import os
import zipfile
from pathlib import Path
import datetime
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2


import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F



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


def get_efficient_net(size='m', pretrained=True):
    print('Входное должно иметь размер 224x224')

    models = {'s': torchvision.models.efficientnet_v2_s(pretrained=pretrained),
              'm': torchvision.models.efficientnet_v2_m(pretrained=pretrained),
              'l': torchvision.models.efficientnet_v2_l(pretrained=pretrained)}

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


class VGG4(nn.Module):
    def __init__(self, num_classes, img_size):
        super(VGG4, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional convolutional layer
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * (img_size // 16) * (img_size // 16), 128)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Additional convolutional layer
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Flatten the data before passing it to fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)

        return x




class InspectorGadjet(nn.Module):
    def __init__(self):
        super(InspectorGadjet, self).__init__()
        
        # Основные сверточные слои
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Для классификации (лицо или фон)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),  # Предположим, что размер входного изображения 96x96
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # Для локализации (ограничивающая рамка лица: x, y, width, height)
        self.regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Преобразуем в 1D тензор
        classification_output = self.classifier(x)
        regression_output = self.regressor(x)
        return classification_output, regression_output


class ConvEmbedding(nn.Module):
    def __init__(self, pic_size=128, emb_size=512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        dims = int(pic_size/32)
        self.emb_size = emb_size
        
        self.fc = nn.Linear(dims*dims*512, self.emb_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class Triplet(nn.Module):
    
    def __init__(self, encoder):
        super(Triplet, self).__init__()
        
        self.encoder = encoder
        
    def forward(self, anchor, pos, neg):
        anchor_embedding = self.encoder(anchor)
        pos_embedding = self.encoder(pos)
        neg_embedding = self.encoder(neg)
        
        return anchor_embedding, pos_embedding, neg_embedding


def combined_loss(pred_class, pred_bbox, target):
    # Разделяем целевой тензор на класс и ограничивающую рамку
    target_class = target[:, 0].float()  # Shape: [batch_size, 1]
    target_bbox = target[:, 1:]  # Shape: [batch_size, 4]

    # Compute the classification loss
    loss_class = F.mse_loss(pred_class.squeeze(), target_class)

    # Compute the regression loss
    loss_bbox = F.smooth_l1_loss(pred_bbox, target_bbox)
    
    # Here, you can assign weights if you want to give different importance to the losses
    combined_loss = loss_class + loss_bbox

    return combined_loss
