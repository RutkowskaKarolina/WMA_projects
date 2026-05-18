import cv2
import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# === SETTINGS ===
model_path = 'fruit_classifier_model_improved.h5'
test_dir = 'dataset/test'
output_video = 'output.mp4'
img_size = (150, 150)
frame_size = (400, 400)
fps = 1

# === Load model ===
model = load_model(model_path)
class_names = list(sorted(os.listdir(test_dir)))  # ['banana', 'lemon', 'orange']

# === Prepare video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

# === Pick samples from each class ===
selected_images = []
for label in class_names:
    folder = os.path.join(test_dir, label)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    sample_size = min(3, len(files))  # max 3 per class
    selected_images.extend(random.sample(files, sample_size))

# === Process each image ===
for image_path in selected_images:
    # Load and preprocess for prediction
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_label = class_names[predicted_idx]

    # Get true label from folder name
    true_label = os.path.basename(os.path.dirname(image_path))

    # Reload image for OpenCV display
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, frame_size)

    # Text overlay
    text = f"Predicted: {predicted_label} ({confidence*100:.1f}%)"
    true_text = f"True: {true_label}"

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, true_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write to video
    video_writer.write(frame)

# === Finalize ===
video_writer.release()
print(f"[✔] Video saved as '{output_video}'")
