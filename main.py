# This is a Rock-paper-scissors dynamic identification code
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np

model = load_model('model_v1.h5')
# 读取视频
vc = cv2.VideoCapture(0)
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False
class_indices = ['paper','rock','scissors']
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret:
        #预测
        pic = cv2.resize(frame[220:420,140:340,:], (500,500))
        name = class_indices[np.argmax(model.predict(pic.reshape(1, 500, 500, 3)))]
        #加工图片
        #边缘检测
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        frame = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        res = cv2.rectangle(frame,(220,140),(420,340),color=(0, 100, 0),thickness=2)
        res = cv2.putText(res, name, (220, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200))
        cv2.imshow('rock-paper-scissors', res)
        if cv2.waitKey(1) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()

# files = os.listdir('./rock-paper-scissors/train/rock')
# image = cv2.imread('./rock-paper-scissors/train/rock/'+files[0])
# pic = cv2.resize(image, (500, 500))
# model.predict(pic.reshape(1, 500, 500, 3))

