import cv2
import matplotlib.pyplot as plt
import os
import time


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor=1.2):
    img_copy = colored_img.copy()

    #Converting to gray
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    #Detecting multiscale  images: some images may be closer to camera than other images
    faces = f_cascade.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0),2)

    return img_copy

#Loading the image
path = 'C:\\Users\\Prawigya\\Desktop\\Projects\\Facial_Recognition\\images'
files=os.listdir(path)
for file in files:
    file='images\\'+file
    test=cv2.imread(str(file))
    #test = convertToRGB(test)
    harr_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    lbp_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    #print("Using Harr Cascade")
    h1=time.time()
    faces_detected_img = detect_faces(harr_face_cascade, test)
    h2=time.time()
    print("Time taken for Harr Cascade for image ",file[7:]," is : ",h2-h1) 
    faces_detected_img=cv2.resize(faces_detected_img, (360,540))

    cv2.imshow(file[7:],faces_detected_img)

    #print("Using LBP Cascade")

    l1=time.time()
    faces_detected_img = detect_faces(lbp_cascade, test)
    l2=time.time()
    print("Time taken for LBP Cascade for image ",file[7:]," is : ",l2-l1)
    print('-'*50)
    print('\n\n')
    





