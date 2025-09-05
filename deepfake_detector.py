import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, 
                                   Dropout, Input, concatenate, BatchNormalization,
                                   GlobalAveragePooling2D, DepthwiseConv2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import dlib
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
from tqdm import tqdm
import gc
import random

# Configure GPU for efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class OptimizedDataProcessor:
    def __init__(self, target_size=(48, 48)):  # Reduced from 64x64 for speed
        self.target_size = target_size
        
    def extract_eyes(self, frame, landmarks):
        """Optimized eye extraction with better error handling"""
        try:
            left_eye_pts = landmarks[36:42]
            right_eye_pts = landmarks[42:48]

            def get_eye_region(points, padding=5):
                x1, y1 = np.min(points[:, 0]) - padding, np.min(points[:, 1]) - padding
                x2, y2 = np.max(points[:, 0]) + padding, np.max(points[:, 1]) + padding
                return max(0, x1), max(0, y1), x2, y2

            lx1, ly1, lx2, ly2 = get_eye_region(left_eye_pts)
            rx1, ry1, rx2, ry2 = get_eye_region(right_eye_pts)

            h, w = frame.shape[:2]
            lx2, ly2 = min(w, lx2), min(h, ly2)
            rx2, ry2 = min(w, rx2), min(h, ry2)

            left_eye = frame[ly1:ly2, lx1:lx2]
            right_eye = frame[ry1:ry2, rx1:rx2]
            
            if left_eye.size == 0 or right_eye.size == 0:
                return None, None
                
            return left_eye, right_eye
        except:
            return None, None

    def calculate_ear(self, eye_points):
        """Eye Aspect Ratio calculation"""
        try:
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C) if C > 0 else 0
        except:
            return 0

    def process_single_video(self, video_info):
        """Process a single video with optimizations"""
        video_path, label, max_frames, frame_skip = video_info
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], [], [], []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, min(total_frames, max_frames * frame_skip), frame_skip))
        random.shuffle(frame_indices)  # Random sampling for better diversity
        frame_indices = frame_indices[:max_frames]
        
        left_imgs, right_imgs, ears, labels = [], [], [], []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                left_img, right_img = self.extract_eyes(frame, landmarks)
                if left_img is None or right_img is None:
                    continue
                    
                # Calculate EAR features
                left_ear = self.calculate_ear(landmarks[36:42])
                right_ear = self.calculate_ear(landmarks[42:48])
                
                # Additional temporal features
                ear_ratio = left_ear / right_ear if right_ear > 0 else 1
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Resize eyes
                left_img = cv2.resize(left_img, self.target_size)
                right_img = cv2.resize(right_img, self.target_size)
                
                # Convert to grayscale for efficiency (optional)
                # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                
                left_imgs.append(left_img)
                right_imgs.append(right_img)
                ears.append([avg_ear, left_ear, right_ear, ear_ratio])
                labels.append(label)
                
                break  # Process only first face per frame for speed

        cap.release()
        return left_imgs, right_imgs, ears, labels

def load_faceforensics_dataset(base_path, max_videos_per_class=500, 
                              max_frames_per_video=50, frame_skip=5, num_workers=None):
    """Load FaceForensics++ dataset with original and manipulated videos"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    processor = OptimizedDataProcessor()
    
    # FaceForensics++ dataset structure
    original_path = os.path.join(base_path, "original_sequences", "youtube", "c23", "videos")
    deepfakes_path = os.path.join(base_path, "manipulated_sequences", "Deepfakes", "c23", "videos")
    
    print(f"Looking for original videos in: {original_path}")
    print(f"Looking for deepfake videos in: {deepfakes_path}")
    
    # Check if paths exist
    if not os.path.exists(original_path):
        print(f"Warning: Original path not found: {original_path}")
        print("You may need to download the original videos using:")
        print("python3 faceforensics_download_v4.py ~/Faceforensics/data -d original -c c23 -t videos --server EU2")
    
    if not os.path.exists(deepfakes_path):
        print(f"Error: Deepfakes path not found: {deepfakes_path}")
        return None, None, None, None
    
    # Prepare video list
    video_list = []
    
    # Real videos (original)
    if os.path.exists(original_path):
        real_videos = [f for f in os.listdir(original_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        real_videos = real_videos[:max_videos_per_class]
        print(f"Found {len(real_videos)} original videos")
        
        for video in real_videos:
            video_list.append((os.path.join(original_path, video), 0, max_frames_per_video, frame_skip))
    else:
        print("Skipping original videos - path not found")
    
    # Fake videos (deepfakes)
    fake_videos = [f for f in os.listdir(deepfakes_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    fake_videos = fake_videos[:max_videos_per_class]
    print(f"Found {len(fake_videos)} deepfake videos")
    
    for video in fake_videos:
        video_list.append((os.path.join(deepfakes_path, video), 1, max_frames_per_video, frame_skip))
    
    if len(video_list) == 0:
        raise ValueError("No videos found! Please check your dataset paths.")
    
    print(f"Processing {len(video_list)} videos with {num_workers} workers...")
    
    # Process videos in parallel
    all_left, all_right, all_ears, all_labels = [], [], [], []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(processor.process_single_video, video_list), 
                          total=len(video_list), desc="Processing videos"))
    
    # Collect results
    for left_imgs, right_imgs, ears, labels in results:
        if len(left_imgs) > 0:
            all_left.extend(left_imgs)
            all_right.extend(right_imgs)
            all_ears.extend(ears)
            all_labels.extend(labels)
    
    if len(all_left) == 0:
        raise ValueError("No valid data found!")
    
    return (np.array(all_left), np.array(all_right), 
            np.array(all_ears), np.array(all_labels))

def create_efficient_model(input_shape=(48, 48, 3), num_ear_features=4):
    """Optimized model architecture using MobileNet-inspired design"""
    
    def depthwise_conv_block(x, filters, kernel_size=3, strides=1):
        x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        return tf.keras.activations.relu(x)
    
    # Left eye branch
    input_left = Input(shape=input_shape)
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_left)
    x1 = BatchNormalization()(x1)
    x1 = depthwise_conv_block(x1, 32, strides=2)
    x1 = depthwise_conv_block(x1, 64, strides=2)
    x1 = GlobalAveragePooling2D()(x1)
    
    # Right eye branch
    input_right = Input(shape=input_shape)
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_right)
    x2 = BatchNormalization()(x2)
    x2 = depthwise_conv_block(x2, 32, strides=2)
    x2 = depthwise_conv_block(x2, 64, strides=2)
    x2 = GlobalAveragePooling2D()(x2)
    
    # EAR features branch
    input_ear = Input(shape=(num_ear_features,))
    x3 = Dense(32, activation='relu')(input_ear)
    x3 = Dropout(0.3)(x3)
    x3 = Dense(16, activation='relu')(x3)
    
    # Combine all branches
    combined = concatenate([x1, x2, x3])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(32, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[input_left, input_right, input_ear], outputs=z)
    
    # Use mixed precision for faster training
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def prepare_data_split(X_left, X_right, X_ear, y, validation_split=0.2):
    """Prepare train/validation split without generators"""
    
    # Convert to proper data types
    X_left = X_left.astype('float32') / 255.0
    X_right = X_right.astype('float32') / 255.0  
    X_ear = X_ear.astype('float32')
    y = y.astype('float32')
    
    # Split data using train_test_split for better randomization
    X_left_train, X_left_val, X_right_train, X_right_val, X_ear_train, X_ear_val, y_train, y_val = train_test_split(
        X_left, X_right, X_ear, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    return (X_left_train, X_right_train, X_ear_train, y_train), (X_left_val, X_right_val, X_ear_val, y_val)

def main():
    # Configuration - Updated for FaceForensics++
    config = {
        'base_path': "/Faceforensics/data",  # Your FaceForensics++ base path
        'max_videos_per_class': 500,  # Limit for faster testing
        'max_frames_per_video': 30,   # Reduced from 100
        'frame_skip': 8,              # Skip frames for speed
        'batch_size': 64,             # Larger batch size
        'epochs': 50,
        'num_workers': mp.cpu_count() // 2
    }
    
    print("Loading FaceForensics++ dataset...")
    try:
        # Try to load cached data first
        with open('faceforensics_processed_data.pkl', 'rb') as f:
            X_left, X_right, X_ear, y = pickle.load(f)
        print("Loaded cached data!")
    except:
        print("Processing FaceForensics++ videos...")
        X_left, X_right, X_ear, y = load_faceforensics_dataset(
            config['base_path'],
            config['max_videos_per_class'], 
            config['max_frames_per_video'],
            config['frame_skip'], 
            config['num_workers']
        )
        
        if X_left is None:
            print("Failed to load dataset. Please check your paths and download the required data.")
            return
        
        # Cache processed data
        with open('faceforensics_processed_data.pkl', 'wb') as f:
            pickle.dump((X_left, X_right, X_ear, y), f)
        print("Data cached for future use!")
    
    print(f"Dataset size: {len(X_left)} samples")
    print(f"Class distribution: Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")
    
    # Prepare train/validation data
    (X_left_train, X_right_train, X_ear_train, y_train), (X_left_val, X_right_val, X_ear_val, y_val) = prepare_data_split(
        X_left, X_right, X_ear, y, validation_split=0.2
    )
    
    print(f"Training samples: {len(X_left_train)}")
    print(f"Validation samples: {len(X_left_val)}")
    
    # Create model
    print("Creating optimized model...")
    model = create_efficient_model()
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks for efficient training
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss'),
        ModelCheckpoint('best_faceforensics_model.keras', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train model
    print("Training...")
    history = model.fit(
        [X_left_train, X_right_train, X_ear_train], y_train,
        validation_data=([X_left_val, X_right_val, X_ear_val], y_val),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Clean up memory
    del X_left, X_right, X_ear, y
    del X_left_train, X_right_train, X_ear_train, y_train
    del X_left_val, X_right_val, X_ear_val, y_val
    gc.collect()
    
    print("Training completed!")
    print("Model saved as 'best_faceforensics_model.keras'")

def predict_video_optimized(video_path, model_path="best_faceforensics_model.keras"):
    """Optimized video prediction"""
    model = tf.keras.models.load_model(model_path)
    processor = OptimizedDataProcessor()
    
    left_imgs, right_imgs, ears, _ = processor.process_single_video(
        (video_path, 0, 50, 5)  # max_frames=50, frame_skip=5
    )
    
    if len(left_imgs) == 0:
        return "No faces detected", 0.0
    
    left_imgs = np.array(left_imgs).astype('float32') / 255.0
    right_imgs = np.array(right_imgs).astype('float32') / 255.0
    ears = np.array(ears).astype('float32')
    
    predictions = model.predict([left_imgs, right_imgs, ears])
    confidence = np.mean(predictions)
    
    result = "Fake" if confidence > 0.5 else "Real"
    return result, confidence

if __name__ == "__main__":
    main()