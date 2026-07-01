# AI-Enabled Helpdesk & Receptionist

An AI-powered, face-recognition-based interactive helpdesk and receptionist application. This system integrates real-time face detection, LBPH (Local Binary Patterns Histograms) face recognition, text-to-speech feedback, and an interactive Streamlit-based kiosk UI to automate guest registration, model training, and custom department inquiries (e.g., college admissions, banking services, attendance marking).

---

## 🚀 Key Features

- **Real-Time Face Detection & Capture**: Utilizes Haar Cascade Classifiers to detect faces and automatically crop, resize (200x200), and normalize lighting before saving training samples.
- **LBPH Facial Recognition**: Trains an efficient Local Binary Patterns Histograms model to identify registered users locally.
- **Text-to-Speech (TTS) Integration**: Uses `pyttsx3` in an isolated background process to prevent Streamlit thread deadlocks while delivering voice greetings and department details.
- **Interactive Kiosk Interface**: Built using Streamlit tabs:
  - 📝 **Register User**: Register a username and unique numeric ID, and capture 100 face images.
  - 🧠 **Train Model**: Train and save the facial recognition model locally.
  - 👁️ **Live Helpdesk**: Detect registered users via webcam, speak personalized greetings, and show dynamic service options.
- **Supportive CLI Utilities**: Includes standalone scripts for CLI-based registration, training, and voice-assisted attendance marking.

---

## 🛠️ Tech Stack & Dependencies

- **Python 3.x**
- **Streamlit** (UI/Dashboard layer)
- **OpenCV (opencv-python, opencv-contrib-python)** (Face detection & recognition)
- **NumPy** (Image matrix operations)
- **Pillow** (Image loading and color conversion)
- **pyttsx3** (Text-to-speech synthesis)

---

## 📦 Directory Structure

```text
AI-Enabled-Helpdesk-main/
├── Data/                              # Directory containing captured training image samples
├── recognizer/                        # Directory where trained LBPH model (.yml) is stored
├── app.py                             # Main Streamlit web application
├── Creater.py                         # CLI script to register new user and capture face data
├── trainer.py                         # CLI script to train the LBPH recognizer model
├── Recognizer.py                      # CLI script for live face detection & attendance voice output
├── haarcascade_frontalface_default.xml # OpenCV pre-trained face detection cascade file
├── user_data.csv                      # Main CSV mapping IDs to Usernames (for Streamlit app)
├── datatext.csv                       # CLI-based registration data mapping
├── requirements.txt                   # Project package dependencies
└── README.md                          # Project documentation
