import os
import cv2
import numpy as np


def data_loader(data_dir):
    image_dir = os.path.join(data_dir, 'files')
    images_data = os.listdir(image_dir)

    labels_file = os.path.join(data_dir, 'labels.txt')
    labels_data = open(labels_file, 'r').readlines()

    images = []
    labels = []

    for i, image in enumerate(images_data):
        img = cv2.imread(os.path.join(image_dir, image))  # read image
        img = cv2.resize(img, (64, 64))     # resize image to the input requirement
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to rgb
        images.append(img)
        labels.append(float(labels_data[i][0]))

    return np.array(images), np.array(labels)
