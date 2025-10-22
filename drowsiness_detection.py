import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import messagebox
    USE_TK = True
except Exception:
    tk = None
    messagebox = None
    USE_TK = False

try:
    import face_recognition
    USE_FACE_RECOG = True
except ImportError:
    face_recognition = None
    USE_FACE_RECOG = False

# Try mediapipe for robust landmark detection if available
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    mp = None
    USE_MEDIAPIPE = False

import platform
if platform.system() == "Windows":
    import winsound
else:
    import subprocess

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

DROWSINESS_FRAMES = 15
EAR_THRESHOLD = 0.23
CAMERA_INDEX = 0
EAR_MEDIAPIPE_THRESHOLD = 0.23

def show_alert_popup():
    if USE_TK and tk and messagebox:
        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showwarning("Drowsiness Alert", "You are sleeping!")
        except Exception:
            print("Drowsiness Alert: You are sleeping!")
        finally:
            root.destroy()
    else:
        print("Drowsiness Alert: You are sleeping!")

def play_alert_sound():
    if platform.system() == "Windows":
        winsound.Beep(2500, 800)
    else:
        subprocess.call(['beep'])

# Log file for drowsiness events
LOG_FILE = 'drowsiness_log.txt'

known_face_encodings = []
known_face_names = []
if USE_FACE_RECOG:
    try:
        img_image = face_recognition.load_image_file('img.jpg')
        img_face_encoding = face_recognition.face_encodings(img_image)[0]
        known_face_encodings = [img_face_encoding]
        known_face_names = ['Authorized Driver']
    except Exception:
        print("Could not load face or no face found in img.jpg. Skipping authentication.")
else:
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_path)
    # prepare eye cascade for fallback eye detection
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    if os.path.exists(eye_cascade_path):
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    else:
        eye_cascade = None

video_capture = cv2.VideoCapture(CAMERA_INDEX)
if not video_capture.isOpened():
    print("Cannot open camera")
    exit()

closed_eyes_frame_count = 0
closed_start_time = None
eye_closed_display_seconds = 0.0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Default values in case no face detected
    status = "No Face"
    color = (0, 255, 255)
    face_locations = []
    face_landmarks_list = []

    # Prefer Mediapipe landmarks if available
    if USE_MEDIAPIPE:
        # Mediapipe expects RGB images
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(rgb_small_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # extract 6 landmarks per eye for EAR using common indices
                    # left eye indices (Mediapipe): [33, 160, 158, 133, 153, 144]
                    # right eye indices: [263, 387, 385, 362, 380, 373]
                    h, w, _ = rgb_small_frame.shape
                    def lm_point(idx):
                        lm = face_landmarks.landmark[idx]
                        return np.array([lm.x * w, lm.y * h])

                    left_pts = np.array([lm_point(i) for i in [33, 160, 158, 133, 153, 144]])
                    right_pts = np.array([lm_point(i) for i in [263, 387, 385, 362, 380, 373]])

                    left_EAR = eye_aspect_ratio(left_pts)
                    right_EAR = eye_aspect_ratio(right_pts)
                    avg_EAR = (left_EAR + right_EAR) / 2.0

                    # Debug print
                    print(f"[MEDIAPIPE] avg_EAR={avg_EAR:.3f}")

                    if avg_EAR < EAR_MEDIAPIPE_THRESHOLD:
                        # eyes considered closed
                        if closed_start_time is None:
                            closed_start_time = time.time()
                        closed_eyes_frame_count += 1
                        eye_closed_display_seconds = time.time() - closed_start_time
                        status = "Sleeping"
                        color = (0, 0, 255)
                    else:
                        # eyes open
                        if closed_start_time is not None:
                            # compute duration and log it
                            duration = time.time() - closed_start_time
                            ts = datetime.now().isoformat(sep=' ', timespec='seconds')
                            entry = f"{ts} - Eyes closed for {duration:.2f} seconds\n"
                            try:
                                with open(LOG_FILE, 'a') as f:
                                    f.write(entry)
                                print(f"[LOG] {entry.strip()}")
                            except Exception as e:
                                print("Failed to write log:", e)
                            closed_start_time = None
                            closed_eyes_frame_count = 0
                            eye_closed_display_seconds = 0.0
                        status = "Awake"
                        color = (0, 255, 0)
                    # draw a label at top-left
                    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    elif USE_FACE_RECOG:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

        for loc, landmarks in zip(face_locations, face_landmarks_list):
            left_eye_pts = np.array([landmarks['left_eye'][i] for i in range(6)])
            right_eye_pts = np.array([landmarks['right_eye'][i] for i in range(6)])

            left_EAR = eye_aspect_ratio(left_eye_pts)
            right_EAR = eye_aspect_ratio(right_eye_pts)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Print EAR for debugging
            print(f"[FACE_RECOG] avg_EAR={avg_EAR:.3f}")

            if avg_EAR < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()
                closed_eyes_frame_count += 1
                eye_closed_display_seconds = time.time() - closed_start_time
                status = "Sleeping"
                color = (0, 0, 255)
            else:
                if closed_start_time is not None:
                    duration = time.time() - closed_start_time
                    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
                    entry = f"{ts} - Eyes closed for {duration:.2f} seconds\n"
                    try:
                        with open(LOG_FILE, 'a') as f:
                            f.write(entry)
                        print(f"[LOG] {entry.strip()}")
                    except Exception as e:
                        print("Failed to write log:", e)
                    closed_start_time = None
                    closed_eyes_frame_count = 0
                    eye_closed_display_seconds = 0.0
                status = "Awake"
                color = (0, 255, 0)

            # Draw box and status label
            top, right, bottom, left = loc
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 40), (right, top), color, -1)
            cv2.putText(frame, status, (left + 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        # fallback: draw yellow box for detected face
        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.rectangle(frame, (left, top - 40), (right, top), (0, 255, 255), -1)
            cv2.putText(frame, "Face", (left + 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Heuristic eye detection using Haar cascade inside face ROI
        eyes_closed_in_frame = 0
        if eye_cascade is not None and len(faces) > 0:
            print(f"[DEBUG] Faces detected: {len(faces)}")
            for fi, (x, y, w, h) in enumerate(faces):
                # small_frame coordinates already used for detection; focus on face ROI
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size == 0:
                    print(f"[DEBUG] Face {fi}: empty ROI_gray, skipping")
                    continue
                detected_eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
                print(f"[DEBUG] Face {fi}: detected_eyes_count={len(detected_eyes)}")
                # For each detected eye, compute height/width ratio; closed eyes typically have smaller height
                for eji, (ex, ey, ew, eh) in enumerate(detected_eyes):
                    ar = float(eh) / float(ew) if ew > 0 else 0.0
                    print(f"[DEBUG] Face {fi} Eye {eji}: ew={ew}, eh={eh}, AR={ar:.3f}")
                    # consider eye closed when ar is small (tuned for small_frame)
                    if ar < 0.22:
                        eyes_closed_in_frame += 1

        # Update closed-eye consecutive frame counter
        if eyes_closed_in_frame >= 1:
            if closed_start_time is None:
                closed_start_time = time.time()
            closed_eyes_frame_count += 1
            eye_closed_display_seconds = time.time() - closed_start_time
            print(f"[DEBUG] eyes_closed_in_frame={eyes_closed_in_frame}, closed_eyes_frame_count={closed_eyes_frame_count}, closed_secs={eye_closed_display_seconds:.2f}")
        else:
            if closed_start_time is not None:
                # log event
                duration = time.time() - closed_start_time
                ts = datetime.now().isoformat(sep=' ', timespec='seconds')
                entry = f"{ts} - Eyes closed for {duration:.2f} seconds\n"
                try:
                    with open(LOG_FILE, 'a') as f:
                        f.write(entry)
                    print(f"[LOG] {entry.strip()}")
                except Exception as e:
                    print("Failed to write log:", e)
                closed_start_time = None
            if closed_eyes_frame_count != 0:
                print(f"[DEBUG] eyes_open -> reset closed_eyes_frame_count (was {closed_eyes_frame_count})")
            closed_eyes_frame_count = 0

        if closed_eyes_frame_count >= DROWSINESS_FRAMES:
            print("ALERT! Driver Drowsiness Detected! (Haar-eye heuristic)")
            threading.Thread(target=play_alert_sound).start()
            threading.Thread(target=show_alert_popup).start()
            closed_eyes_frame_count = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
print("Program exited successfully.")
