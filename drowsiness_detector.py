import cv2
import mediapipe as mp
import numpy as np
import time
import csv

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.25
CLOSED_EYES_TIME = 2  # seconds

start_time = None
alert_active = False

def log_alert(event):
    with open("alerts.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Left eye landmarks
            left = [33, 160, 158, 133, 153, 144]

            eye = []
            for idx in left:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                eye.append((int(lm.x * w), int(lm.y * h)))

            # EAR calculation
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

            ear = (A + B) / (2.0 * C)

            if ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                else:
                    if time.time() - start_time > CLOSED_EYES_TIME:
                        alert_active = True
                        log_alert("DROWSINESS")
            else:
                start_time = None
                alert_active = False

    if alert_active:
        frame[:] = (0, 0, 255)
        cv2.putText(frame, "DROWSY!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()