import datetime
import os
import time
import cv2
import pandas as pd

def recognize_attendance():
    print("[INFO] Current Working Directory:", os.getcwd())

    model_path = "./TrainingImageLabel/Trainner.yml"
    cascade_path = "haarcascade_frontalface_default.xml"
    student_details_path = "StudentDetails/StudentDetails.csv"

    # === Check required files ===
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at {model_path}")
        return
    if not os.path.exists(cascade_path):
        print(f"[ERROR] Haar cascade file not found at {cascade_path}")
        return
    if not os.path.exists(student_details_path):
        print(f"[ERROR] StudentDetails.csv not found at {student_details_path}")
        return

    # === Load resources ===
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    faceCascade = cv2.CascadeClassifier(cascade_path)
    df = pd.read_csv(student_details_path)

    print("[INFO] Model and student details loaded successfully.")

    # === Initialize ===
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Department', 'BatchYear', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("[INFO] Starting video capture. Press 'q' to quit.")

    while True:
        ret, im = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image from camera.")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,
                                             minSize=(int(minW), int(minH)),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            confidence_percent = round(100 - conf)

            print(f"[DEBUG] Predicted Id: {Id}, Raw Confidence: {conf:.2f}, Confidence%: {confidence_percent}%")

            # Better match = lower conf value
            if conf < 60:
                student_row = df.loc[df['Id'] == Id]
                if not student_row.empty:
                    name = student_row.iloc[0]['Name']
                    dept = student_row.iloc[0].get('Department', 'N/A')
                    batch = student_row.iloc[0].get('BatchYear', 'N/A')
                else:
                    name, dept, batch = f"User {Id}", "Unknown", "Unknown"

                color = (0, 255, 0)
                label_text = f"{name} ({dept}, {batch})"
                conf_text = f"{confidence_percent}%"

                # Record attendance
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                attendance.loc[len(attendance)] = [Id, name, dept, batch, date, timeStamp]
            else:
                name, dept, batch = "Unknown", "N/A", "N/A"
                color = (0, 165, 255)
                label_text = name
                conf_text = f"{confidence_percent}%"

            # Draw face box and text
            cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
            cv2.putText(im, label_text, (x + 5, y - 10), font, 0.8, color, 2)
            cv2.putText(im, conf_text, (x + 5, y + h - 10), font, 0.8, color, 1)

        # Avoid duplicate IDs
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Recognize & Attendance', im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Save Attendance File ===
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')

    if not os.path.exists("Attendance"):
        os.makedirs("Attendance")

    fileName = f"Attendance/Attendance_{date}_{timeStamp}.csv"
    attendance.to_csv(fileName, index=False)
    print(f"[INFO] Attendance saved successfully: {fileName}")

    cam.release()
    cv2.destroyAllWindows()
