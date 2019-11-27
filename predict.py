#! -*- coding:utf-8 -*-0

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
 
import matplotlib.pyplot as plt
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import time


import tensorflow as tf
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
 
# 设置tensorflow session 为Keras 后端
keras.backend.tensorflow_backend.set_session(get_session())


def file_is_img(img_path):
    postfix = img_path.strip().split('.')[-1]
    if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
        if postfix in ['JPG', 'JPEG']:
            os.rename(img_path, img_path.replace(postfix, 'jpg'))
            print(img_path)
        return True
    else:
        return False

def get_image_list(dir_path):
    img_list = []
    if not os.path.exists(dir_path):
        return []
    for root, dirs, files in os.walk(dir_path):
        for img in files:
            img_path = os.path.join(root, img)
            if file_is_img(img_path):
                img_list.append(img_path)
    return img_list

def main():
    classes = ["hat", "person"]
    labels_to_names = {0:"hat", 1:"person"}
    #加载模型
    model_path ='./model/resnet50_csv_03.h5'
    model = models.load_model(model_path, backbone_name='resnet50')
    img_dir = './images'
    for img_path in get_image_list(img_dir): 
        image = read_image_bgr(img_path)   
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        image = preprocess_image(image)
        image, scale = resize_image(image)
        # 模型预测
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        # 矫正比例
        boxes /= scale
        # 目标检测可视化展示
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # 设置预测得分最低阈值
            if score < 0.50:
                break
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        #图片展示
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        new_img_path = img_path.replace('images', 'images/result')
        plt.savefig(new_img_path,format='png',transparent=True,pad_inches=0,dpi=300,bbox_inches='tight')

if __name__ == '__main__':
    main()
