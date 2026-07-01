import streamlit as st
import cv2
import numpy as np
import pyttsx3
import os
import csv
from PIL import Image
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

import sys
import subprocess

def speak_async(msg):
    """Speaks text using pyttsx3 in an isolated background process to prevent Streamlit thread deadlocks."""
    safe_msg = msg.replace('"', '\\"').replace('\n', ' ')
    script = f"""import pyttsx3
engine = pyttsx3.init()
engine.say("{safe_msg}")
engine.runAndWait()
"""
    # Run the speech in a completely separate background process
    subprocess.Popen([sys.executable, "-c", script])

# Helper function to read user data from CSV
def get_user_data():
    user_data = {}
    if os.path.exists("user_data.csv"):
        with open("user_data.csv", mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:
                    user_data[row[0]] = row[1]
    return user_data

# Helper function to write user data to CSV
def save_user_data(user_id, username):
    with open("user_data.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, username])

st.set_page_config(page_title="AI-Enabled Helpdesk", layout="wide")
st.title("🤖 AI-Enabled Helpdesk")

# Create tabs for the 3 main functions
tab1, tab2, tab3 = st.tabs(["📝 Register User", "🧠 Train Model", "👁️ Live Helpdesk"])

# --- TAB 1: REGISTER USER ---
with tab1: 
    st.header("Register New User")
    
    with st.form("registration_form", clear_on_submit=True):
        username = st.text_input("Enter Username:")
        user_id = st.text_input("Enter Unique ID (Numbers only):")
        start_registration = st.form_submit_button("Start Registration")
    
    if start_registration:
        if not username or not user_id.isdigit():
            st.error("Please enter a valid username and a numeric ID.")
        else:
            current_users = get_user_data()
            if str(user_id) in current_users or username in current_users.values():
                st.error(f"User '{username}' or ID '{user_id}' is already registered! Please go to the Live Helpdesk.")
                speak_async("You have already registered. Please go to the live help desk.")
            else:
                if not os.path.exists("Data"):
                    os.makedirs("Data")
                
                save_user_data(user_id, username)
                st.success(f"User {username} saved! Look into the webcam...")
                speak_async("Please look into the camera.")
                
                # Start camera
                cap = cv2.VideoCapture(0)
                FRAME_WINDOW = st.image([])
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                val = 0
            while val < 100:
                status, img = cap.read()
                if not status:
                    st.error("Error: Could not access the webcam.")
                    break
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    val += 1
                    
                    # 15% inner crop to strictly remove background
                    margin_y = int(h * 0.15)
                    margin_x = int(w * 0.15)
                    face_img = gray[y + margin_y : y + h - margin_y, x + margin_x : x + w - margin_x]
                    
                    # Standardize dimensions
                    face_img = cv2.resize(face_img, (200, 200))
                    
                    cv2.imwrite(f"Data/{user_id}.{val}.jpg", face_img)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Update Progress
                    progress_bar.progress(val / 100)
                    status_text.text(f"Captured {val}/100 frames")
                
                # Show Feed
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(img_rgb)
                
            cap.release()
            st.success("Face data captured successfully! Go to the 'Train Model' tab next.")

# --- TAB 2: TRAIN MODEL ---
with tab2:
    st.header("Train AI Model")
    st.write("Train the facial recognition model using the captured face data.")
    
    if st.button("Start Training"):
        if not os.path.exists("Data") or len(os.listdir("Data")) == 0:
            st.error("No training data found in the 'Data' folder. Please register a user first.")
        else:
            if not os.path.exists("recognizer"):
                os.makedirs("recognizer")
                
            # Use default grid, but keep crop/resize improvements
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            path = 'Data'
            
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')] 
            faces = []
            users = []
            
            with st.spinner("Training in progress..."):
                for img_path in image_paths:
                    face_img = Image.open(img_path).convert('L')
                    face_np = np.array(face_img, 'uint8')
                    face_np = cv2.equalizeHist(face_np) # Normalize lighting
                    curr_id = int(os.path.split(img_path)[-1].split('.')[0])
                    
                    faces.append(face_np)
                    users.append(curr_id)
            
            if len(faces) > 0:
                recognizer.train(faces, np.array(users))
                recognizer.save('recognizer/TrainingData.yml')
                st.success("✅ Training complete and saved successfully!")
            else:
                st.error("Failed to process training data.")

# --- TAB 3: LIVE HELPDESK ---
with tab3:
    st.header("Live Helpdesk Recognizer")
    
    if 'current_guest' not in st.session_state:
        st.session_state['current_guest'] = None
    if 'guest_info' not in st.session_state:
        st.session_state['guest_info'] = ""
    if 'kiosk_category' not in st.session_state:
        st.session_state['kiosk_category'] = 'main'

    if st.session_state['current_guest'] is None:
        run_helpdesk = st.checkbox("Turn On Webcam")
        
        if run_helpdesk:
            if not os.path.exists("recognizer/TrainingData.yml"):
                st.error("Training data file not found! Please train the model first.")
            else:
                rec = cv2.face.LBPHFaceRecognizer_create()
                rec.read("recognizer/TrainingData.yml")
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                user_data = get_user_data()
                
                import time
                import threading
                
                # Store camera in session state to guarantee release
                if 'cap' not in st.session_state:
                    st.session_state['cap'] = None
                    
                if st.session_state['cap'] is None:
                    st.session_state['cap'] = cv2.VideoCapture(0)
                    
                cap = st.session_state['cap']
                FRAME_WINDOW = st.image([])
                
                st.warning("Webcam is active. Looking for a registered face...")
                
                start_time = time.time()
                last_unknown_spoken = 0
                consecutive_recognitions = 0
                consecutive_unknowns = 0
                last_recognized_name = ""
                
                try:
                    while run_helpdesk:
                        status, img = cap.read()
                        if not status:
                            continue

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            
                            # Match the strict 15% crop from training
                            margin_y = int(h * 0.15)
                            margin_x = int(w * 0.15)
                            face_roi = gray[y + margin_y : y + h - margin_y, x + margin_x : x + w - margin_x]
                            
                            # Match the 200x200 resize
                            face_roi = cv2.resize(face_roi, (200, 200))
                            
                            face_roi = cv2.equalizeHist(face_roi) # Normalize lighting
                            id_pred, conf = rec.predict(face_roi)

                            if conf > 65:  
                                name = "Unknown"
                            else:
                                name = user_data.get(str(id_pred), "Unknown")

                            # Put text on image
                            cv2.putText(img, f"{name} ({round(conf,1)})", (x, y - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                            # Show Feed so they see the bounding box before transition
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(img_rgb)
                            
                            if name != "Unknown":
                                consecutive_unknowns = 0
                                if name == last_recognized_name:
                                    consecutive_recognitions += 1
                                else:
                                    last_recognized_name = name
                                    consecutive_recognitions = 1
                                    
                                if consecutive_recognitions >= 15:
                                    # Instantly release camera to avoid light staying on
                                    cap.release()
                                    st.session_state['cap'] = None
                                    
                                    # Transition to Kiosk Mode
                                    st.session_state['current_guest'] = name
                                    st.session_state['greeted'] = False
                                    st.rerun()
                            else:
                                consecutive_recognitions = 0
                                if time.time() - start_time > 3: # 3 second grace period at startup
                                    consecutive_unknowns += 1
                                    if consecutive_unknowns >= 30: # Requires ~1 second of solid Unknown
                                        if time.time() - last_unknown_spoken > 5:
                                            speak_async("You haven't registered. Please register first.")
                                            last_unknown_spoken = time.time()

                        # Show Feed if no face transitioned
                        if cap.isOpened():
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(img_rgb)
                finally:
                    if cap is not None and cap.isOpened():
                        cap.release()
                        st.session_state['cap'] = None
        else:
            # Guarantee camera is released when unchecked
            if 'cap' in st.session_state and st.session_state['cap'] is not None:
                st.session_state['cap'].release()
                st.session_state['cap'] = None
    else:
        # KIOSK MODE (RECEPTIONIST UI)
        guest_name = st.session_state['current_guest']
        
        if not st.session_state.get('greeted', False):
            speak_async(f"Hi {guest_name}, what are you looking for?")
            st.session_state['greeted'] = True
            
        st.success(f"### 👋 Welcome, {guest_name}!")
        
        category = st.session_state['kiosk_category']
        
        if category == 'main':
            st.write("How can I help you today? Please select an option below:")
            col1, col2, col3 = st.columns(3)
            
            if col1.button("🏫 College Services", use_container_width=True):
                st.session_state['kiosk_category'] = 'college'
                st.session_state['guest_info'] = ""
                st.rerun()
                
            if col2.button("🏦 Banking Services", use_container_width=True):
                st.session_state['kiosk_category'] = 'banking'
                st.session_state['guest_info'] = ""
                st.rerun()
                
            if col3.button("❌ End Session", use_container_width=True, type="primary"):
                speak_async("Goodbye, have a great day!")
                st.session_state['current_guest'] = None
                st.session_state['guest_info'] = ""
                st.session_state['greeted'] = False
                st.session_state['kiosk_category'] = 'main'
                st.rerun()
                
        elif category == 'college':
            st.write("🏫 **College Services**")
            col1, col2, col3, col4 = st.columns(4)
            
            if col1.button("🎓 Admissions", use_container_width=True):
                st.session_state['guest_info'] = "**Admissions Department:**\n\nAdmissions for the upcoming Fall semester are currently open. You can apply online or visit the admin block (Building A, Room 101) for physical forms. The deadline is August 15th."
                speak_async("Admissions for the upcoming Fall semester are currently open. You can apply online or visit the admin block in Building A for physical forms. The deadline is August 15th.")
                
            if col2.button("💼 Placements", use_container_width=True):
                st.session_state['guest_info'] = "**Placements & Career Services:**\n\nOur college has a 95% placement rate! Upcoming campus drives include Google, Microsoft, and TCS next month. Visit the Career Center in Building C for resume reviews."
                speak_async("Our college has a 95 percent placement rate! Upcoming campus drives include Google, Microsoft, and TCS next month. Visit the Career Center in Building C for resume reviews.")
                
            if col3.button("🏢 Facilities", use_container_width=True):
                st.session_state['guest_info'] = "**Campus Facilities:**\n\n- **Library:** Open 24/7 during exams.\n- **Cafeteria:** Main food court is in the central plaza.\n- **Sports Complex:** Includes a gym, indoor stadium, and swimming pool."
                speak_async("Our campus facilities include a 24 7 Library during exams, a main food court in the central plaza, and a sports complex with a gym, indoor stadium, and swimming pool.")
                
            if col4.button("⬅️ Back to Main", use_container_width=True):
                st.session_state['kiosk_category'] = 'main'
                st.session_state['guest_info'] = ""
                st.rerun()
                
        elif category == 'banking':
            st.write("🏦 **Banking Services**")
            col1, col2, col3, col4 = st.columns(4)
            
            if col1.button("💳 Open Account", use_container_width=True):
                st.session_state['guest_info'] = "**Open an Account:**\n\nWe offer zero-balance student accounts and premium savings accounts. To open an account, please provide your ID proof and a passport-sized photograph to Counter 2."
                speak_async("We offer zero balance student accounts and premium savings accounts. To open an account, please provide your I D proof and a photograph to Counter 2.")
                
            if col2.button("💰 Loans", use_container_width=True):
                st.session_state['guest_info'] = "**Loan Inquiries:**\n\nWe offer educational loans at an interest rate of 6% and personal loans at 10%. Please meet our loan officer in Cabin 3 for detailed EMI calculations."
                speak_async("We offer educational loans at 6 percent and personal loans at 10 percent. Please meet our loan officer in Cabin 3 for detailed calculations.")
                
            if col3.button("🏧 Credit Cards", use_container_width=True):
                st.session_state['guest_info'] = "**Credit Card Applications:**\n\nApply for our Rewards Credit Card with zero annual fee for the first year! Minimum required credit score is 700."
                speak_async("Apply for our Rewards Credit Card with zero annual fee for the first year! The minimum required credit score is 700.")
                
            if col4.button("⬅️ Back to Main", use_container_width=True):
                st.session_state['kiosk_category'] = 'main'
                st.session_state['guest_info'] = ""
                st.rerun()
                
        if st.session_state['guest_info']:
            st.info(st.session_state['guest_info'])
