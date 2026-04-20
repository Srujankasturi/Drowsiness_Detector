import cv2
import mediapipe as mp
import numpy as np
import time
import csv

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.25
CLOSED_EYES_TIME = 3  # seconds
TILT_THRESHOLD = 50  # tune this for your face

start_time = None
alert_active = False
nod_alert = False

def log_alert(event):
    with open("alerts.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event])


def get_ear(face_landmarks, frame, indices):
    eye = []
    h, w, _ = frame.shape

    for idx in indices:
        lm = face_landmarks.landmark[idx]
        eye.append((int(lm.x * w), int(lm.y * h)))

    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)


def get_head_tilt(landmarks, w, h):
    nose = landmarks[1]
    chin = landmarks[152]
    nose_y = int(nose.y * h)
    chin_y = int(chin.y * h)
    return chin_y - nose_y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    nod_alert = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Left eye landmarks
            left  = [33, 160, 158, 133, 153, 144]
            right = [362, 385, 387, 263, 373, 380]

            left_ear = get_ear(face_landmarks, frame, left)
            right_ear = get_ear(face_landmarks, frame, right)
            tilt = get_head_tilt(face_landmarks.landmark, w, h)

            cv2.putText(frame, f"EAR: {(left_ear + right_ear) / 2:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tilt: {tilt}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                else:
                    if time.time() - start_time > CLOSED_EYES_TIME:
                        alert_active = True
                        log_alert("DROWSINESS")
            else:
                start_time = None
                alert_active = False

            if tilt < TILT_THRESHOLD:
                nod_alert = True
                log_alert("HEAD_NOD")
            else:
                nod_alert = False

    if alert_active or nod_alert:
        frame[:] = (0, 0, 255)
        cv2.putText(frame, "DROWSY!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()