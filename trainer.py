import os
import cv2
import numpy as np
from PIL import Image

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'Data'  # Folder containing training images

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')] 
    faces = []
    users = []

    for img_path in image_paths:
        face_img = Image.open(img_path).convert('L')  # Convert to grayscale
        face_np = np.array(face_img, 'uint8')  # Convert image to numpy array
        user_id = int(os.path.split(img_path)[-1].split('.')[0])  # Extract ID from filename
        
        faces.append(face_np)
        users.append(user_id)

        print(f"Processing user ID: {user_id}")

        cv2.imshow("Training", face_np)
        cv2.waitKey(100)

    return users, faces

# Load images and labels
users, faces = get_images_and_labels(path)

# Train only if faces are detected
if len(faces) > 0:
    recognizer.train(faces, np.array(users))
    recognizer.save('recognizer/TrainingData.yml')  # Save trained model
    print("Training complete and saved as 'TrainingData.yml'")
else:
    print("No training data found. Please check the dataset.")

cv2.destroyAllWindows()
