# python azure2/train.py --data-folder ./data
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

from azureml.core import Run

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=256,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=256,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
n_epochs=10
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)

X_train = np.load(os.path.join(data_folder,"features.npy"))
y_train = np.load(os.path.join(data_folder,"labels.npy"))
X_test = X_train[:20]
y_test = y_train[:20]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

model = Sequential()

model.add(Conv2D(n_h1, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(n_h2, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=['accuracy'])

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])


history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    validation_split=0.3,
                    callbacks=[LogRunMetrics()])

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('Dog/Cat Classification ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
