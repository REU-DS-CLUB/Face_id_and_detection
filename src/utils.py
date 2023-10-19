import pandas as pd
import yaml
from datetime import datetime
from comet_ml import Experiment
from pathlib import Path

import cv2
from numpy import inf
import torchvision
import torchvision.transforms as tf
import torch


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
    # !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

    # !kaggle datasets download -d sbaghbidi/human-faces-object-detection
    # !mkdir data
    # !mv human-faces-object-detection.zip data

    data_path = Path("data/")
    image_path = data_path / "human-faces-object-detection"
    with zipfile.ZipFile(data_path / "human-faces-object-detection.zip", "r") as zip_ref:
        print("Unzipping...")
        zip_ref.extractall(image_path)

    image_path = data_path / "human-faces-object-detection" / "images"

    y_labels = pd.read_csv(
        '/content/data/human-faces-object-detection/faces.csv')

    images_paths = list(image_path.glob('*.jpg'))

    return image_path, y_labels


def get_background_dataset_for_colab():
    from google.colab import files

    uploaded = files.upload()

    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))

    # Then move kaggle.json into the folder where the API expects to find it.
    # !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

    # !kaggle datasets download -d sbaghbidi/human-faces-object-detection
    # !mkdir data
    # !mv human-faces-object-detection.zip data

    data_path = Path("data/")
    zip_name = "house_room_dataset"
    image_path = data_path / zip_name
    with zipfile.ZipFile(data_path / f"{zip_name}.zip", "r") as zip_ref:
        print("Unzipping...")
        zip_ref.extractall(image_path)

    image_path = data_path / zip_name

    return image_path


def save_img(img, pred, epoch):
    have_face = pred[0]
    if have_face: 
        box = torchvision.utils.draw_bounding_boxes(
            img, [pred[1], pred[2], pred[3], pred[4]], colors='red')
        pil_image = torchvision.transforms.ToPILImage()(box)
        image_path = f"./results/epoch_{epoch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pil_image.save(image_path)


def rescale_coordinates(tensor, original_shape=(720, 1280), model_shape=(126, 126)):
    # Извлекаем координаты из тензора
    x1, y1, x2, y2 = tensor[0][1:]

    # Масштабирование координат
    x1 = int((x1 / model_shape[1]) * original_shape[1])
    y1 = int((y1 / model_shape[0]) * original_shape[0])
    x2 = int((x2 / model_shape[1]) * original_shape[1])
    y2 = int((y2 / model_shape[0]) * original_shape[0])

    return [x1, y1, x2, y2]


def cam_capture(source=0, model=None, bbox_func=None, limit=inf):

    """""
    source - источник видео, если 0,то это камера ноутбука
    model - модель, выдающая координаты
    bbox_func - функция, котороая строит изображение с bounding box'ом на основе предиктов модели
    limit - количество милисекунд, в течение которых работает камера

    """""

    cap = cv2.VideoCapture(source)
    i = 0 

    while i<=limit:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        
        # Преобразуем изображение из BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Изменяем размер изображения до 126x126
        resized_frame = cv2.resize(rgb_frame, (128, 128))
        
        

        pic_tens = tf.ToTensor()(resized_frame)
        
        

        with torch.no_grad():

            res = model(pic_tens.unsqueeze(0))
        
        coord = rescale_coordinates(res)
      

        cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
        
        print(coord)
        cv2.imshow("Camera Feed with BBox", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()