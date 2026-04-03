# app.py
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from scipy import interpolate
from flask import Flask, request, render_template, redirect, url_for

# ----------------------------
# 1. Flask setup
# ----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------------
# 2. Load TFLite model
# ----------------------------
model_path = "model15.tflite"  # path to your TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# 3. Load index -> sign mapping
# ----------------------------
with open("sign_to_prediction_index_mapp.json", "r", encoding="utf-8") as f:
    sign_map = json.load(f)
    sign_map = {int(v): k for k, v in sign_map.items()}

# ----------------------------
# 4. Mediapipe setup
# ----------------------------
mp_pose = mp.solutions.pose

# ----------------------------
# 5. Helper functions
# ----------------------------
def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                landmarks_list.append(landmarks)

    cap.release()
    return np.array(landmarks_list)

def prepare_input_for_tflite(landmarks, target_frames=543):
    if landmarks.shape[0] == 0:
        raise ValueError("No landmarks detected in the video.")

    xyz = landmarks[:, :, :3]
    flattened = xyz.reshape(xyz.shape[0], -1)
    f = interpolate.interp1d(np.arange(flattened.shape[0]), flattened, axis=0)
    resized = f(np.linspace(0, flattened.shape[0]-1, target_frames))
    input_data = resized[:, :3]
    input_data = np.expand_dims(input_data, axis=0)
    return input_data.astype(np.float32)

def predict_sign(video_path, top_k=5):
    landmarks = extract_landmarks_from_video(video_path)
    input_data = prepare_input_for_tflite(landmarks)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index']).flatten()

    # Top-K predictions
    top_indices = output_data.argsort()[-top_k:][::-1]
    top_predictions = [(sign_map.get(idx, "Unknown"), float(output_data[idx])) for idx in top_indices]

    return top_predictions

# ----------------------------
# 6. Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No file part"
        file = request.files["video"]
        if file.filename == "":
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            predictions = predict_sign(filepath, top_k=5)
            return render_template("result.html", predictions=predictions)
    return render_template("index.html")

# ----------------------------
# 7. Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
