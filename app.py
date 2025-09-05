import streamlit as st
import tempfile
import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import dlib

# ===============================
# Load Model & Face Detector
# ===============================
model = load_model("best_faceforensics_model.keras")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ===============================
# Helper Functions
# ===============================
def extract_eyes(frame, landmarks):
    left_eye_pts = landmarks[36:42]
    right_eye_pts = landmarks[42:48]

    def crop_eye(eye):
        x1, y1 = np.min(eye[:, 0]), np.min(eye[:, 1])
        x2, y2 = np.max(eye[:, 0]), np.max(eye[:, 1])
        return frame[y1:y2, x1:x2]

    return crop_eye(left_eye_pts), crop_eye(right_eye_pts)

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def process_uploaded_video(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    left_imgs, right_imgs, ears = [], [], []
    frame_ids = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            try:
                left_img, right_img = extract_eyes(frame, landmarks)
                left_ear = calculate_ear(landmarks[36:42])
                right_ear = calculate_ear(landmarks[42:48])
                avg_ear = (left_ear + right_ear) / 2.0

                # Resize to match model input (48x48)
                left_img = cv2.resize(left_img, (48, 48)).astype('float32') / 255.0
                right_img = cv2.resize(right_img, (48, 48)).astype('float32') / 255.0

                left_imgs.append(left_img)
                right_imgs.append(right_img)
                ears.append([left_ear, right_ear, avg_ear, abs(left_ear - right_ear)])
                frame_ids.append(count)
            except:
                continue

        count += 1

    cap.release()
    if len(left_imgs) == 0:
        return None, None, None, None
    return np.array(left_imgs), np.array(right_imgs), np.array(ears), frame_ids

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🕵️ Deepfake Video Detector")
st.markdown("Upload a video and let AI analyze whether it's **Real or Fake** on a per-frame basis.")

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_path = temp_video.name

    with st.spinner("⏳ Processing video..."):
        left, right, ears, frame_ids = process_uploaded_video(temp_path)

    if left is None:
        st.error("No faces detected in the video. Try another one.")
    else:
        st.success(f"✅ Processed {len(left)} frames successfully!")

        with st.spinner("🔍 Running prediction..."):
            preds = model.predict([left, right, ears])
            avg_score = float(np.mean(preds))
            result = "Fake" if avg_score > 0.5 else "Real"

        # --- Metrics Summary ---
        st.subheader("📊 Detection Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames Analyzed", len(left))
        col2.metric("Avg Fake Score", f"{avg_score*100:.2f}%")
        col3.metric("Final Verdict", "🚨 Fake" if result == "Fake" else "✅ Real")

        # --- Tabs for Results ---
        tabs = st.tabs(["🎯 Final Prediction", "📈 Per-frame Chart", "📥 Download Results"])

        with tabs[0]:
            st.markdown("### 🎯 Final Prediction")
            if result == "Fake":
                st.error(f"🚨 FAKE video with {avg_score*100:.2f}% confidence.")
            else:
                st.success(f"✅ REAL video with {(1 - avg_score)*100:.2f}% confidence.")

        with tabs[1]:
            st.markdown("### 📊 Per-frame Prediction Chart")
            result_df = pd.DataFrame({
                "Frame": frame_ids,
                "Prediction Score": preds.flatten()
            }).set_index("Frame")

            st.line_chart(result_df, height=400)
            st.caption("Scores above 0.5 indicate **Fake** frames.")

        with tabs[2]:
            st.markdown("### 🧾 Export Results")
            result_df = pd.DataFrame({
                "Frame": frame_ids,
                "Score": preds.flatten(),
                "Label": ["Fake" if p > 0.5 else "Real" for p in preds.flatten()]
            })

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv, file_name="prediction_results.csv", mime="text/csv")

        os.remove(temp_path)
