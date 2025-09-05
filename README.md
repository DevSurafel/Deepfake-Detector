# Deepfake Video Detector

A user-friendly web application for detecting deepfake videos using deep learning and eye-blink analysis. The app leverages facial landmark detection and a custom-trained CNN model to analyze video frames and predict whether a video is real or fake.

## Project Description

This project provides a Streamlit-based interface for deepfake detection in videos. Users can upload a video file, and the app will:

- Detect faces and extract eye regions from each frame using dlib.
- Calculate eye aspect ratio (EAR) and other features to analyze blinking patterns.
- Use a trained deep learning model to predict the likelihood of each frame being fake.
- Aggregate per-frame predictions to provide an overall verdict for the video.
- Visualize results with charts and metrics.
- Allow users to download detailed per-frame prediction results as a CSV file.

The detection pipeline combines computer vision (OpenCV, dlib) and deep learning (TensorFlow/Keras) for robust and explainable deepfake analysis.

## How It Works

1. **Face & Landmark Detection:** Each frame is processed to detect faces and extract 68 facial landmarks.
2. **Eye Region Extraction:** The left and right eye regions are cropped and resized for model input.
3. **Feature Calculation:** EAR and related features are computed for each eye.
4. **Prediction:** The model predicts the probability of each frame being fake.
5. **Result Aggregation:** The app summarizes results, visualizes per-frame scores, and provides a final verdict.

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/DevSurafel/Deepfake-Detector.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download required model and landmark files:**
    - `best_faceforensics_model.keras`
    - `shape_predictor_68_face_landmarks.dat`

4. **Run the app:**
    ```bash
    streamlit run app.py
    ```

5. **Upload a video file** (`.mp4`, `.avi`, `.mov`) and view results in your browser.

## Notes

- **Do not upload FaceForensics++ datasets or large video files to this repository.**
- Model weights and landmark files should be downloaded separately (provide links or instructions).

## License

MIT
