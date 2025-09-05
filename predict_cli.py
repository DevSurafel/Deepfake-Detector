from tensorflow.keras.models import load_model
from deepfake_detector import predict_video

model = load_model("best_faceforensics_model.keras")

video_path = input("Enter path to video: ")
result = predict_video(video_path, model)

print(f"\nPrediction: {result}")
