import cv2

def camer():
    print("\n[INFO] Initializing camera... Press 'q' to quit anytime.\n")

    # Load Haar Cascade for face detection
    cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print(f"[ERROR] Could not load Haar Cascade from {cascade_path}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam. Check your camera settings.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)

        # Show live camera feed
        cv2.imshow('Camera Check - Press Q to Exit', img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Camera test closed successfully.")
            break

    cap.release()
    cv2.destroyAllWindows()
