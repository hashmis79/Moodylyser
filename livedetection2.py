import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.utils import np_utils
from sklearn.utils import shuffle
import keras

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model('Moodel1')#Loading the pre-trained model

cap = cv2.VideoCapture(0)#starting the VideoCapture
data2 = []
while True:
    ret, frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.6, 4)#detecting faces using face_cascade
    cropped = frame[0:2,0:2]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-100,y-100), ((x+w+100), (y+h+100)), (255, 0, 0), 3)#created a rectangle around the face
        roi_gray = gray[y:y+h,x:x+w]
        cropped = gray[y-100:y+h+100,x-100:x+w+100]#cropped out the face area for better detection by the model
        roi_color = frame[y:y+h,x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex,ey), ((ex+ew), (ey+eh)), (0, 255, 0), 3)
    if(cropped.shape < (10,10,3)):
        print('Not detected/out of frame')
    else:
        print(cropped.shape)
        testimg = cv2.resize(cropped,(48,48))#processed the image to be of the size (48,48,1)
        print(frame.shape)
        data2 = np.array(testimg)/255.0#converted the image to a numpy array
        print(data2.shape)
        data2 = np.reshape(data2,(1,48 ,48,1))
        X = model.predict(data2)[0]#Used the model to predict the emotion
        print(X)#Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        list = X.tolist()
        maxpos = list.index(max(list))
        if maxpos == 0 :#different conditions for the emotion
            cv2.putText(frame, "Status : {}".format('Angry'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif maxpos == 1 :
            cv2.putText(frame, "Status : {}".format('Disgust'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif maxpos == 2 :
            cv2.putText(frame, "Status : {}".format('Fear'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif maxpos == 3 :
            cv2.putText(frame, "Status : {}".format('Happy'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif maxpos == 4 :
            cv2.putText(frame, "Status : {}".format('Sad'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif maxpos == 5 :
            cv2.putText(frame, "Status : {}".format('Surprise'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif maxpos == 6 :
            cv2.putText(frame, "Status : {}".format('Neutral'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        cv2.imshow('frame',frame)
        cv2.imshow('testimg',testimg)
    key = cv2.waitKey(1) %256
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
