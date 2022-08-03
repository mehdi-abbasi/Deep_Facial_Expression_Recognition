from tabnanny import verbose
import cv2
from keras.models import load_model
from tensorflow import keras
import numpy as np


def face_extractor(image, faces):
    face_images = []
    for face in faces:
        x,y,w,h = face
        face_pixels = image[y:y+h,x:x+w]
        face_pixels = cv2.resize(face_pixels, (48,48), interpolation = cv2.INTER_AREA)
        face_images.append(face_pixels)

    return np.uint16(face_images)


model = load_model('best_model.h5')
haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')

# TODO: CHECL LABELS
label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}

cap = cv2.VideoCapture(0)
while True:
    _,img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    face_images = face_extractor(gray_img,faces_rect)
    if face_images.any():
        preds = model.predict(face_images,verbose=0)
        y_pred = np.argmax(preds , axis = 1 )

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label_dict[int(y_pred[0])], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Detected Face Image',  img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break