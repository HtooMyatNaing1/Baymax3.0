import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download

# Load the face mask detection model from Hugging Face
st.write("🔄 Loading model...")
model_path = hf_hub_download(repo_id="kaungkhantcoder/FaceMaskDetection", filename="mask_detection_model.h5")
model = load_model(model_path)
st.success("✅ Model Loaded Successfully!")

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("🖥️ Face Mask Detection")
st.write("🔹 Click 'Start Camera' to begin detection.")

# Start and Stop buttons
start_button = st.button("▶ Start Camera")
stop_button = st.button("⏹ Stop Camera")

# Placeholder for video feed
video_placeholder = st.empty()

# Use session state to manage webcam
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start_button:
    st.session_state.camera_active = True

if stop_button:
    st.session_state.camera_active = False

# Open webcam if the camera is active
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)  # Automatically detect webcam

    if not cap.isOpened():
        st.error("⚠️ Unable to access webcam. Please check your camera settings.")
    else:
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Cannot read from camera. Try restarting your device.")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]  # Crop face
                face = cv2.resize(face, (150, 150))  # Resize for model
                face = img_to_array(face) / 255.0  # Normalize
                face = np.expand_dims(face, axis=0)

                # Predict mask or no mask
                prediction = model.predict(face)[0][0]
                label = "Mask" if prediction < 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()
