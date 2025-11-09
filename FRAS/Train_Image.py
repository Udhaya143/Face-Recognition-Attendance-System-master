import os
import time
import cv2
import numpy as np
from PIL import Image


# ------------------ Function to Get Images and Labels ------------------
def getImagesAndLabels(path):
    imagePaths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    faces = []
    Ids = []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')  # Grayscale
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract ID
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"[WARNING] Skipping invalid image: {imagePath} ({e})")
            continue

    return faces, Ids


# ------------------ Train Images Function ------------------
def TrainImages():
    training_path = "TrainingImage"
    model_folder = "TrainingImageLabel"
    model_path = os.path.join(model_folder, "Trainner.yml")

    # Check if TrainingImage folder exists
    if not os.path.exists(training_path):
        print("[ERROR] 'TrainingImage' folder does not exist! Please capture faces first.")
        return

    faces, Ids = getImagesAndLabels(training_path)

    if len(faces) == 0:
        print("[ERROR] No valid training images found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print(f"\n[INFO] Starting training on {len(faces)} images...")

    # Train recognizer
    recognizer.train(faces, np.array(Ids))

    # Ensure model folder exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save model
    recognizer.save(model_path)

    # Simulate progress for user feedback
    total = len(faces)
    for i in range(total + 1):
        percent = int((i / total) * 100)
        print(f"\r[INFO] Training Progress: {percent}% Complete", end="")
        time.sleep(0.03)

    print("\n✅ [SUCCESS] Training completed 100% and model saved successfully!")
    print(f"📁 Model Location: {model_path}")
