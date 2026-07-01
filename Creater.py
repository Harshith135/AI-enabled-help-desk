import cv2
import numpy as np
import os

# Load Haar cascade for face detection
load = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# User details input
name = input("Enter Username: ")
id = input("Enter Unique id (maxLen = 4): ")

# Create Data directory if not exists
if not os.path.exists("Data"):
    os.makedirs("Data")

# Save user details
with open("datatext.csv", "a") as f:
    f.write(str(id) + " " + name + "\n")

val = 0  # Image count

while True:
    status, img = cap.read()
    if not status:
        print("Error: Could not access the webcam")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = load.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        val += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"Data/{id}.{val}.jpg", face_img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.waitKey(50)

    cv2.imshow('FaceDetect', img)
    if cv2.waitKey(1) & 0xFF == ord('q') or val >= 50:
        break

cap.release()
cv2.destroyAllWindows()





