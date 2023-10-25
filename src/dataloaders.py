import torch

import zipfile
from pathlib import Path

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pathlib import Path

from tqdm import tqdm
import datetime
import random
import cv2
import src.utils as utils

import albumentations as A


config = utils.get_options()

if config['use_colab']:
    image_path, y_labels = utils.get_detection_dataset_for_colab()
    backg_image_path = utils.get_background_dataset_for_colab()
else:
    y_labels = pd.read_csv('data/data_detection/faces.csv')
    image_path = 'data/data_detection/images'
    backg_image_path = 'data/data_background'


class FacesDataset(Dataset):

    def __init__(self, images_path, dataset, transform, height=64, width=64):
        ''' Loading dataset
        images_path: path where images are stored
        dataset: dataframe where image names and box bounds are stored
        transform: functions for data augmentation (from albumentations lib)
        height: height used to resize the image
        width: width used to resize the image
        images_list: list where all image paths are stored
        bboxes: list where all the bounding boxes are stored
        '''
        self.images_path = Path(images_path)
        self.dataset = dataset
        self.height = height
        self.width = width
        self.n_samples = dataset.shape[0]

        self.images_list = sorted(list(self.images_path.glob('*.jpg')))
        self.images_names = [image.name for image in self.images_list]
        self.bboxes_names = dataset['image_name'].tolist()

        self.transform = transform

   # cut down to only images present in dataset

        self.images = []
        for i in self.bboxes_names:
            for j in self.images_names:
                if i == j:
                    self.images.append(i)

    def __getitem__(self, index):

        image_name = self.images[index]
        image_path = self.images_path / image_name

        img = cv2.imread(str(image_path))
        # by default in cv2 represents image in BGR order, so we have to convert it back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height)).astype(np.float32)
        img /= 255.0  # normalizing values
        img = np.transpose(img, (2, 0, 1))  # converting to CHW format

        image_labels = self.dataset[self.dataset['image_name'] == image_name]

        imgs, bbox = [], []
        for i in range(len(image_labels)):
            cur_height = image_labels['height'].iloc[i]
            cur_width = image_labels['width'].iloc[i]

            x0 = (image_labels['x0'].iloc[i] / cur_width) * self.width
            y0 = (image_labels['y0'].iloc[i] / cur_height) * self.height
            x1 = (image_labels['x1'].iloc[i] / cur_width) * self.width
            y1 = (image_labels['y1'].iloc[i] / cur_height) * self.height

            bbox = torch.tensor([1, x0, y0, x1, y1]).float()
            break

        items = self.transform(image=np.transpose(img, (1, 2, 0)), bboxes=[list(bbox[1:])], class_labels=[1])
        img = np.transpose(items['image'], (2, 0, 1)) # converting back to CHW format

        if len(items['bboxes']) > 0:
            bbox = torch.tensor([1] + list(items['bboxes'][0]))
        else:
            bbox = [0, -1, -1, -1, -1] # if bbox is too small after the augmentation we drop the bbox

        return img, bbox

    def __len__(self):
        return self.n_samples


class BackgroundDataset(Dataset):

    def __init__(self, folder_path, transform, height=64, width=64):
        ''' Loading dataset
        folder_path: path of images of background
        '''
        self.folder_path = Path(folder_path)
        self.height = height
        self.width = width
        self.transform = transform

        self.types = ['Bathroom', 'Bedroom', 'Dinning', 'Kitchen', 'Livingroom']

        images_path = []
        for type in self.types:
            type_folder = os.path.join(folder_path, type)
            images = os.listdir(type_folder)
            images_path += [os.path.join("data", os.path.relpath(os.path.join(type_folder, img), 'data/'))
                            for img in images]
        random.shuffle(images_path)
        self.images_path = images_path[:2500]

        self.n_samples = len(self.images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height)).astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))

        return img, torch.tensor([0, -1, -1, -1, -1]).float()

    def __len__(self):
        return self.n_samples

transform_faces = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5),
    A.Flip(p=0.5),
    A.GaussianBlur(p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))

transform_obj = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([transforms.RandomRotation(degrees=30), 
                            transforms.ColorJitter(brightness=0.2, contrast=0.2),
                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

batch_size = config['batch_size']
img_size = config['img_size']
dataset_for_detection = FacesDataset(image_path, y_labels, transform_faces, img_size, img_size)
dataset_of_backgrounds = BackgroundDataset(backg_image_path, transform_obj, img_size, img_size)

dataset = ConcatDataset([dataset_for_detection, dataset_of_backgrounds])
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
