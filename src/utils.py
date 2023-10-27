import pandas as pd
import yaml
from datetime import datetime
from comet_ml import Experiment
from pathlib import Path
import subprocess

import cv2
from numpy import inf
import torchvision
import torchvision.transforms as tf
import torch
import zipfile
import os
import shutil
import csv


def get_options():
    options_path = 'config.yaml'
    with open(options_path, 'r') as option_file:
        options = yaml.safe_load(option_file)
    return options


def get_kaggle_json_file():
    from google.colab import files
    print('ЗАГРУЗИТЕ kaggle.json ФАЙЛ')

    uploaded = files.upload()

    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))


def execute_terminal_comands(commands):
  for command in commands:
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def download_dataset_from_kaggle(full_name_of_dataset, name_of_dataset ):

    print(f'\nНачинаю скачачивать датасет: {name_of_dataset}')

    download_face_detection_dataset = [
        f"kaggle datasets download -d {full_name_of_dataset}",
        "mkdir data",
        f"mv {name_of_dataset}.zip data/"
        ]
    
    execute_terminal_comands(download_face_detection_dataset)

    data_path = Path("data/")
    image_path = data_path / name_of_dataset
    with zipfile.ZipFile(data_path / f"{name_of_dataset}.zip", "r") as zip_ref:
        print(f"Unzipping dataset {name_of_dataset}")
        zip_ref.extractall(image_path)


def download_datasets_from_kaggle():

    get_kaggle_json_file()

    # перемещение файла kaggle json в место, где его ожидает библиотека kaggle чтоб скачать датасет
    move_kaggle_json_file = [
        "mkdir -p ~/.kaggle/",
        "mv kaggle.json ~/.kaggle/",
        "chmod 600 ~/.kaggle/kaggle.json"
    ]
    execute_terminal_comands(move_kaggle_json_file)

    # Скачать датасет с лицами 10к
    download_dataset_from_kaggle('fareselmenshawii/face-detection-dataset', 'face-detection-dataset')

    # Скачать датасет с комнатами
    download_dataset_from_kaggle('robinreni/house-rooms-image-dataset', 'house-rooms-image-dataset')

    # Скачать датасет с лицами 3к
    download_dataset_from_kaggle('sbaghbidi/human-faces-object-detection', 'human-faces-object-detection')
    


def preprocessing_of_face_detection_dataset():
    print('Начинаю обработку датасета face_detection_dataset')

    def move_all_files(name):
        # Пути к папкам train и val
        train_path = f"/content/Face_id_and_detection/data/face-detection-dataset/{name}/train"
        val_path = f"/content/Face_id_and_detection/data/face-detection-dataset/{name}/val"

        # Функция для перемещения файлов из source_dir в target_dir
        def move_files(source_dir, target_dir):
            # Получаем список файлов в исходной директории
            files = os.listdir(source_dir)

            # Перемещаем каждый файл в целевую директорию
            for file_name in files:
                source_file = os.path.join(source_dir, file_name)
                target_file = os.path.join(target_dir, file_name)
                shutil.move(source_file, target_file)

        # Перемещаем файлы из папки train в /content/data/face-detection-dataset/labels/
        move_files(train_path, f"/content/Face_id_and_detection/data/face-detection-dataset/{name}/")

        # Перемещаем файлы из папки val в /content/data/face-detection-dataset/labels/
        move_files(val_path, f"/content/Face_id_and_detection/data/face-detection-dataset/{name}/")

        # Удаляем пустые папки train и val
        os.rmdir(train_path)
        os.rmdir(val_path)

    for name in ['labels', 'images']:
        move_all_files(name)
        print(f'done with {name}')

    print('starting last section')
        # Папки с файлами .txt и папка с изображениями
    labels2_dir = "/content/Face_id_and_detection/data/face-detection-dataset/labels2"


    # Путь к итоговому CSV файлу
    csv_file_path = "/content/Face_id_and_detection/data/face-detection-dataset/labels_and_coordinates.csv"

    # Открываем CSV файл для записи
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['name', 'x1', 'y1', 'x2', 'y2']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Записываем заголовки CSV файла
        writer.writeheader()

        # Проходим по каждому файлу .txt в папке labels2
        for filename in os.listdir(labels2_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(labels2_dir, filename)

                # Открываем файл .txt для чтения
                with open(file_path, 'r') as txt_file:
                    lines = txt_file.readlines()

                    # Проверяем количество строк в файле
                    if len(lines) <= 2:
                        # Получаем имя файла без расширения
                        name = filename[:-4]

                        # Получаем x1, y1, x2, y2 из первой строки
                        parts = lines[0].split()
                        if len(parts) == 6 and parts[0] == "Human" and parts[1] == "face":
                            x1, y1, x2, y2 = map(float, parts[2:])

                            # Записываем данные в CSV файл
                            writer.writerow({'name': name, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    print(f"Данные записаны в {csv_file_path}")

def colab():

    download_datasets_from_kaggle()

    preprocessing_of_face_detection_dataset()

    print('done with colab')



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


import torchvision.transforms as tf

def crop(pic, coords, scale=2, size=256):
    
    '''
    pic - входное изображение PIL
    coords - координаты рамки
    scale - фактор скалирования рамки (то, во сколько раз обрезанное изображение больше рамки)
    size - размер выходного изображение
    
    '''
    
    pic_width = pic.size[0]
    pic_height = pic.size[1]
        
    
    center = (0.5*(coords[2]+coords[0]), 
              0.5*(coords[3]+coords[1]))
    
    width = scale*(coords[2]-coords[0])
    height = scale*(coords[3]-coords[1])
    
    side = min(max(width, height), min(pic_width, pic_height))
    
    bot_right = [min(center[0]+side/2, pic_width), 
                 min(center[1]+side/2, pic_height)]
    
    x0 = max(0, bot_right[0] - side)
    y0 = max(0, bot_right[1] - side)
    
    res = tf.functional.resized_crop(pic, y0, x0, min(side, pic_height), min(side, pic_width), size=size)
    
    return res, center
