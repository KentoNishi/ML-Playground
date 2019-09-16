import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import keras
from tqdm import tqdm
import random

dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"dataset\data")
print(dir)
classes = ["Dog", "Cat"]
resolution=100

def datasetExists():
    return (os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),"./features.npy")) and \
        os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),"./labels.npy")))

if(datasetExists()==False):
    training_data=[]
    for category in classes:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.join(dir,category))
        class_num = classes.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (resolution, resolution))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    print(len(training_data))
    random.shuffle(training_data)
    X = []
    Y = []
    for features,label in training_data:
        X.append(features)
        Y.append(label)
    X = np.array(X).reshape(-1, resolution, resolution, 1)
    X = X/255.0
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"features.npy"), X)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"labels.npy"), Y)
# https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/?completed=/loading-custom-data-deep-learning-python-tensorflow-keras/