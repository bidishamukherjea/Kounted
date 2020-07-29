import cv2
import sys
import matplotlib.pyplot as plt
import os
from scipy import ndimage, misc
# Get user supplied values

directory = os.fsdecode('./database')
total = 0
#image = cv2.imread('C:/Users/BIDISHA/Desktop/FaceCount/the_party.jpg',1)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        # Read the image
        path = os.path.join(directory,filename)
        image = cv2.imread(path,1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        total += len(faces)
        print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


print("Total detected faces = {}".format(total))
        #status = cv2.imwrite(filename, image)
#print ("Image written to file-system : ",status)
# cv2.waitKey(0)