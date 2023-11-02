import os
from pathlib import Path
import random

import pandas as pd
import numpy as np
import cv2
import albumentations as A

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import src.utils as utils

# загружаем данные из конфига
config = utils.get_options()

batch_size = config['batch_size']
img_size = config['img_size']
img_size_recog = config['img_size_recog']




# задаем корневую папку
root = '/content/Face_id_and_detection/' if config['use_colab'] else ""

# пути к папкам датасета с 3к изображениям лиц
y_labels = pd.read_csv(f'{root}data/human-faces-object-detection/faces.csv')
image_path = f'{root}data/human-faces-object-detection/images'

# путь к папке с картинками комнат
backg_image_path = f'{root}data/house-rooms-image-dataset/House_Room_Dataset'

# Путь к папке с изображениями и CSV файлу для датасета с 10к изображениями
image_dir_for_ten_thousand_dataset = f"{root}data/face-detection-dataset/images"
csv_file_path_for_ten_thousand_dataset = f"{root}data/face-detection-dataset/labels_and_coordinates.csv"


### ЗАДАЕМ КЛАСССЫ ДАТАСЕТОВ ###

class ThreeThousandFaceDataSet(Dataset):

    def __init__(self, images_path, dataset, transform=None, transform_bbox=None):
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
        self.n_samples = dataset.shape[0]

        self.images_list = sorted(list(self.images_path.glob('*.jpg')))
        self.images_names = [image.name for image in self.images_list]
        self.bboxes_names = dataset['image_name'].tolist()

        self.transform_bbox = transform_bbox
        self.transform = transform

        #img size from config
        self.size = img_size

        # cut down to only images present in dataset
        self.images = []
        for i in self.bboxes_names:
            for j in self.images_names:
                if i == j:
                    self.images.append(i)


    def __getitem__(self, index):

        # Получение изображения как массива numpy 
        image_name = self.images[index]
        image_path = self.images_path / image_name
        img = cv2.imread(str(image_path))


        # by default in cv2 represents image in BGR order, so we have to convert it back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # извлекаем ширину и высоту текущего изображения
        cur_height, cur_width = img.shape[:2]

        # img = img.astype(float) / 255.0   # custom normalization
        
        image_labels = self.dataset[self.dataset['image_name'] == image_name]

        for i in range(len(image_labels)):
            # cur_height = image_labels['height'].iloc[i]
            # cur_width = image_labels['width'].iloc[i]

            x0 = (int(image_labels['x0'].iloc[i]) / cur_width) * self.size
            y0 = (int(image_labels['y0'].iloc[i]) / cur_height) * self.size
            x1 = (int(image_labels['x1'].iloc[i]) / cur_width)  * self.size
            y1 = (int(image_labels['y1'].iloc[i]) / cur_height) * self.size

            bbox = np.array([1, x0, y0, x1, y1])
            break

        # at this point we have img as RGB like np.array not normalized and
        # bbox as np.array

        # resize изображение, чтобы применить albumentations
        img = cv2.resize(img, (self.size, self.size))

        if self.transform_bbox:
            items = self.transform_bbox(image=img, bboxes=[bbox[1:]], class_labels=[1])
            img = items['image'] 

            if len(items['bboxes']) > 0:
                bbox = [1] + list(items['bboxes'][0])
                # bbox = list(items['bboxes'][0])
            else:
                # if bbox is too small after the augmentation we drop the bbox
                bbox = [0, -1, -1, -1, -1]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(bbox)

    def __len__(self):
        return self.n_samples


class BackgroundDataset(Dataset):

    def __init__(self, folder_path, transform = None):
        ''' Loading dataset
        folder_path: path of images of background
        '''
        self.folder_path = Path(folder_path)
        self.size = img_size
        self.transform = transform

        self.types = ['Bathroom', 'Bedroom',
                      'Dinning', 'Kitchen', 'Livingroom']
        
        images_path = []

        for type in self.types:
            type_folder = os.path.join(folder_path, type)
            images = os.listdir(type_folder)
            images_path += [os.path.join("data", os.path.relpath(os.path.join(type_folder, img), 'data/'))
                            for img in images]
            
        random.shuffle(images_path)

        self.images_path = images_path

        self.n_samples = len(self.images_path)

    def __getitem__(self, index):
        # Получение изображения как массива numpy 
        image_path = self.images_path[index]
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.size, self.size))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([0, -1, -1, -1, -1]).float()

    def __len__(self):
        return self.n_samples


class TenThousandFaceDataSet(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, transform_bbox=None):
        # csv file with bbox values
        self.data = pd.read_csv(csv_file)

        #path to dir with images
        self.image_dir = Path(image_dir)

        #torch transforms that we put in init
        self.transform = transform
        self.transform_bbox = transform_bbox

        #img size from config
        self.size = img_size
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # Получение изображения как массива numpy 
        img_name = os.path.join(self.image_dir.joinpath(f"{self.data.iloc[idx, 0]}"))  
        img = cv2.imread(str(img_name)+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Извлекаем координаты bbox из CSV файла
        x1, y1, x2, y2 = self.data.iloc[idx, 1:5].values.astype(np.float32)

        # извлекаем ширину и высоту текущего изображения
        cur_height, cur_width = img.shape[:2]

        # Создаем ограничивающий прямоугольник (bbox)
        bbox = np.array([1, x1/cur_width*self.size, y1/cur_height*self.size, x2/cur_width*self.size, y2/cur_height*self.size])
        
        ### at this point we have img as RGB like np.array not normalized and
        ### bbox as np.array

        # img = img.astype(float) / 255.0   # custom normalization
        
        img = cv2.resize(img, (self.size, self.size))

        # apply transform for bbox if needed
        if self.transform_bbox is not None:
            items = self.transform_bbox(img=np.transpose(img, (1, 2, 0)), bboxes=[
                                        bbox[1:]], class_labels=[1])
            # img = np.transpose(items['image'], (2, 0, 1)) # converting back to HHWC format
            print(items)
            if len(items['bboxes']) > 0:
                bbox = [1] + list(items['bboxes'][0])
            else:
                # if bbox is too small after the augmentation we drop the bbox
                bbox = [0, -1, -1, -1, -1]


        # Применяем преобразования к изображению (если указаны)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(bbox)


class CelebATriplets(Dataset):
    def __init__(self, images, triplets_path, transform=None):
        self.images_path = Path(images)
        self.triplets_path = Path(triplets_path)
        self.triplets = pd.read_csv(self.triplets_path)
        self.transform = transform
        self.size = img_size_recog
        

    def __getitem__(self, index):
        triplet = self.triplets[self.triplets.index == index]

        def get_img(image_path):
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size)).astype(np.float32)
            img /= 255.0
            img = np.transpose(img, (2, 0, 1))
            return img

        anc_path = Path(triplet.anchor.values[0])
        pos_path = Path(triplet.pos.values[0])
        neg_path = Path(triplet.neg.values[0])

        anc = get_img(self.images_path.joinpath(anc_path))
        pos = get_img(self.images_path.joinpath(pos_path))
        neg = get_img(self.images_path.joinpath(neg_path))

        # pos_id = triplet.id2.values[0]
        # neg_id = triplet.id3.values[0]
        return [anc, pos, neg]

    def __len__(self):
        return len(self.triplets)



### TRANSFORMS SECTION ###
transform_faces = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.5, p=0.7),
    A.GaussianBlur(p=0.01)
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['class_labels'])) # min_area=1024 min_visibility=0.1

#transform for each img 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0), # normalization
])


### INITIALISING DATASETS ###

# --Detection Dataloader--

#dataset for 3000 img with faces
Ten_Thousand_Face_dataset = TenThousandFaceDataSet(
    csv_file=csv_file_path_for_ten_thousand_dataset, 
    image_dir=image_dir_for_ten_thousand_dataset, 
    transform=transform, transform_bbox=None)

#dataset for 10000 img with faces
Three_Thousand_Face_dataset = ThreeThousandFaceDataSet(image_path, y_labels, 
                                                       transform, transform_bbox=transform_faces)

#dataset with backgrounds img
dataset_of_backgrounds = BackgroundDataset(backg_image_path, transform)

#concatinating all datasets
dataset = ConcatDataset(
    [Three_Thousand_Face_dataset, Ten_Thousand_Face_dataset, dataset_of_backgrounds])


detection_dataloader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True)


# --FaceId Dataloader--
# celebA dataset
celeb_images = f"{root}data/CelebA FR Triplets/images"
celeb_triplets_csv = f"{root}data/CelebA FR Triplets/triplets.csv"

CelebA_dataset = CelebATriplets(celeb_images, celeb_triplets_csv)

recognition_dataloader = DataLoader(
    dataset=CelebA_dataset, batch_size=batch_size, shuffle=True)
