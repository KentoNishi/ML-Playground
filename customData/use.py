import cv2
import tensorflow as tf
import sys
import os

if(len(sys.argv)>1):
    classes = ["Dog", "Cat"]
    def prepare(filepath):
        resolution = 100
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (resolution, resolution))
        return new_array.reshape(-1, resolution, resolution, 1)

    model = tf.keras.models.load_model("model.h5")
    prediction = model.predict([prepare(os.path.join(os.path.dirname(__file__),sys.argv[1]))])
    print(classes[int(prediction[0][0])])
else:
    print("Please specify a relative file path.")