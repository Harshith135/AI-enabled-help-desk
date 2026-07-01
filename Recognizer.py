import cv2
import numpy as np
import pyttsx3
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load face recognizer
if not os.path.exists("recognizer/TrainingData.yml"):
    print("Error: Training data file not found!")
    exit()

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/TrainingData.yml")

# Font for text display
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Load user data
user = {}
if os.path.exists("datatext.txt"):
    with open("datatext.txt", "r") as f:
        for line in f:
            line = line.strip()
            # Handle cases where line has more than 2 items or empty lines
            if line and len(line.split()) == 2:
                key, value = line.split(" ")
                user[key] = value
            else:
                print(f"Skipping invalid line: {line}")

while True:
    status, img = cap.read()
    if not status:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])  # Face recognition

        print("Detected ID:", id, "Confidence:", conf)

        if conf > 75:  # Confidence threshold
            name = "Unknown"
        else:
            name = user.get(str(id), "Unknown")

        # Construct a custom message
        if name != "Unknown":
            message = f"Hi {name} from KIIS ROBOTIC & AI, marking your attendance as present"
        else:
            message = "Hi, I don't recognize you marking absent."

        # Text-to-speech output
        engine.say(message)
        engine.runAndWait()

        # Display name on frame
        cv2.putText(img, name, (x, y + h), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FaceDetect', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
