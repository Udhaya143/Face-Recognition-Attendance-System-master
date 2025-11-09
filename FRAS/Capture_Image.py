import cv2
import os
import pandas as pd
import time

def takeImages():
    dataset_path = "TrainingImage"
    student_details_path = "StudentDetails/StudentDetails.csv"

    # Ensure folders exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists("StudentDetails"):
        os.makedirs("StudentDetails")

    # Ask for ID first
    Id = input("Enter Numeric User ID (e.g., 1): ")

    # Load or create CSV
    if os.path.exists(student_details_path):
        df = pd.read_csv(student_details_path)
    else:
        df = pd.DataFrame(columns=["Id", "Name", "Department", "Batch_Year"])

    # Check if ID already exists
    if int(Id) in df["Id"].values:
        existing_user = df.loc[df["Id"] == int(Id)].iloc[0]
        name = existing_user["Name"]
        department = existing_user["Department"]
        batch_year = existing_user["Batch_Year"]

        print(f"\n[INFO] Existing user found: {name} ({department}, {batch_year})")
        print("[INFO] Department and Batch Year not required again.")
    else:
        name = input("Enter your Name (e.g., John): ")
        department = input("Enter your Department (e.g., Computer Science): ")
        batch_year = input("Enter your Batch Year (e.g., 2025): ")

        df.loc[len(df)] = [int(Id), name, department, batch_year]
        df.to_csv(student_details_path, index=False)
        print(f"[INFO] Details saved successfully to {student_details_path}")

    # Initialize face classifier
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)  # Width
    cam.set(4, 720)   # Height

    print(f"\n[INFO] Capturing faces for {name} ({department}, {batch_year})")
    print("[INFO] Look at the camera. Press 'q' to quit early.")

    sample_count = 0
    max_samples = 50  # Capture 50 images for better accuracy

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Cannot access camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            sample_count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            file_path = os.path.join(dataset_path, f"User.{Id}.{sample_count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Captured {sample_count}/{max_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Capturing Faces', img)

        if sample_count >= max_samples or cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] {sample_count} images captured successfully for ID {Id} ({name}).")
