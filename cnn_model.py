import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import image
from tqdm import tqdm
import os
import numpy as np
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# temp = []
# files = os.listdir('./rock-paper-scissors/train/rock')
# for file in files:
#     image = cv2.imread('./rock-paper-scissors/train/rock/'+file)
#     print(image.shape)
#     temp.append(image.shape)
train = ImageDataGenerator(rescale=1 / 255)
test = ImageDataGenerator(rescale=1 / 255)
train_generator = train.flow_from_directory(
    './rock-paper-scissors/train',
    target_size=(500,500),
    batch_size=50,
    class_mode='sparse'
)
test_generator = test.flow_from_directory(
    './rock-paper-scissors/test',
    target_size=(500,500),
    batch_size=50,
    class_mode='sparse'
)

# 搭建模型
model = keras.Sequential()
# 两层卷积
model.add(keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(500,500,3)))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.softmax))
# 训练与测试
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
model.save('model_v1.h5')
