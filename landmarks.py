import numpy as np
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# import keras
import dlib
import tensorflow as tf

model = tf.keras.models.load_model('Moodelld1_5de.h5')

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
frontalface_detector = dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
width=0
height=0
counters = [0.0,0.0,0.0,0.0,0.0]
counter = 0;
while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[0:2,0:2]
    faces = frontalface_detector(frame, 1)
    for (i, face) in enumerate(faces):#Applying Landmarks on the cropped frame
        (x, y, w, h) = rect_to_bb(face)
        black = (0, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image = create_blank(frame.shape[1],frame.shape[0], rgb_color=black)
        if len(faces):
            landmarks = [(p.x, p.y) for p in landmark_predictor(frame, faces[0]).parts()]
        radius = -1
        circle_thickness = 2
        for (x1, y1) in landmarks:
            cv2.circle(gray, (x1, y1), circle_thickness, (0,255,0), radius)
        roi_gray = gray[y-20:y+h+20,x-50:x+w+20]

    testimg = cv2.resize(roi_gray,(48,48))#processed the image to be of the size (48,48,1)
    print(frame.shape)
    data2 = np.array(testimg)/255.0#converted the image to a numpy array
    print(data2.shape)
    data2 = np.reshape(data2,(1,48 ,48,1))
    X = model.predict(data2)[0]#Used the model to predict the emotion
    print(X)#Negative{Angry', 'Disgust', 'Fear'}, 'Happy', 'Sad', 'Surprise', 'Neutral'
    list = X.tolist()
    maxpos = list.index(max(list))
    if maxpos == 0 :#different conditions for the emotion
        cv2.putText(frame, "Status : {}".format('Negative Emotion'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        counters[0] = counters[0]+1
        counter = counter +1
    elif maxpos == 1 :
        cv2.putText(frame, "Status : {}".format('Happy'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        counters[1] = counters[1]+1
        counter = counter +1
    elif maxpos == 2 :
        cv2.putText(frame, "Status : {}".format('Sad'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        counters[2] = counters[2]+1
        counter = counter +1
    elif maxpos == 3 :
        cv2.putText(frame, "Status : {}".format('Surprise'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        counters[3] = counters[3]+1
        counter = counter +1
    elif maxpos == 4 :
        cv2.putText(frame, "Status : {}".format('Neutral'),(10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        counters[4] = counters[4]+1
        counter = counter +1
    cv2.imshow('black',roi_gray)
    cv2.imshow('frame',frame)#Displaying the frames

    key = cv2.waitKey(1) %256
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(counters)
print(counter)
for i in range(5):
    counters[i] = counters[i]*100/counter#Calculatng Percentages of emotions
print(counters)

classes_list = ['Neg', 'hap', 'sad', 'sur', 'neu']
plt.plot(classes_list, counters)
plt.xlabel('Emotions')
plt.ylabel('Percentage')
plt.show()#Displaying the Graph of Percentage vs Emotions

key = cv2.waitKey(1) %256
if key & 0xFF == ord('q'):
    cv2.destroyAllWindows()
