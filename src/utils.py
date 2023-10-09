import pandas as pd
import yaml
from datetime import datetime
from comet_ml import Experiment
from pathlib import Path

import cv2 
from numpy import inf
import torchvision
import torchvision.transforms as tf

def get_options():
    options_path = 'config.yaml'
    with open(options_path, 'r') as option_file:
        options = yaml.safe_load(option_file)
    return options


def get_detection_dataset_for_colab():
    from google.colab import files

    uploaded = files.upload()

    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))

    # Then move kaggle.json into the folder where the API expects to find it.
    !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

    !kaggle datasets download -d sbaghbidi/human-faces-object-detection
    !mkdir data
    !mv human-faces-object-detection.zip data

    data_path = Path("data/")
    image_path = data_path / "human-faces-object-detection"
    with zipfile.ZipFile(data_path / "human-faces-object-detection.zip", "r") as zip_ref:
        print("Unzipping...")
        zip_ref.extractall(image_path)

    image_path = data_path / "human-faces-object-detection" / "images"

    y_labels = pd.read_csv('/content/data/human-faces-object-detection/faces.csv')

    images_paths = list(image_path.glob('*.jpg'))

    return image_path, y_labels


def save_img(img, pred, epoch):
    box = torchvision.utils.draw_bounding_boxes(img, [pred[0], pred[1], pred[2], pred[3]], colors = 'red')
    torchvision.transforms.ToPILImage()(box)
    image_path = f"./results/epoch_{epoch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pil_image.save(image_path)


def cam_capture(source=0, model=None, bbox_func=None, limit=inf):

    """""
    source - источник видео, если 0,то это камера ноутбука
    model - модель, выдающая координаты
    bbox_func - функция, котороая строит изображение с bounding box'ом на основе предиктов модели
    limit - количество милисекунд, в течение которых работает камера

    """""

    cam = cv2.VideoCapture(source)
    i = 0 

    while i<=limit:

        pic = cam.read()[1]

        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic_tens = tf.ToTensor()(pic)

        res = model(pic_tens)

        bboxed = bbox_func(pic.uint8(), res.uint8(), colors = 'red') ## функция tv.utils.draw_bounding_boxes() принимает на вход только uint8 тензоры

        output = cv2.cvtColor(bboxed.permute([1, 2, 0]).numpy(), cv2.COLOR_BGR2RGB)

        cv2.imshow('Detection output', output)
        cv2.waitKey(1)

        i += 1
    
    cam.release()