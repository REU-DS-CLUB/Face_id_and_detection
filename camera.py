import cv2 
from numpy import inf

import torchvision.transforms as tf


def cam_capture(source=0, model=None, bbox_func=None, limit=inf):

    """""
    source - источник видеоБ если 0,то это камера ноутбука
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