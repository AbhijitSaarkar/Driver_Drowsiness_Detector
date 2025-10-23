import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
from collections import deque
import platform

# Optional libraries with graceful fallbacks
try:
    import tkinter as tk
    from tkinter import messagebox
    USE_TK = True
except:
    USE_TK = False

try:
    import face_recognition
    USE_FACE_RECOG = True
except:
    USE_FACE_RECOG = False

try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
except:
    USE_MEDIAPIPE = False

# USB Relay Control (Windows)
try:
    if platform.system() == "Windows":
        import winsound
        import serial
        USE_USB_RELAY = True
    else:
        import subprocess
        USE_USB_RELAY = False
except:
    USE_USB_RELAY = False

# ==================== CONFIGURATION ====================
# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection thresholds (research-validated)
EAR_THRESHOLD = 0.21           # Eye Aspect Ratio for closed eyes
MAR_THRESHOLD = 0.5            # Mouth Aspect Ratio for yawning
HEAD_TILT_THRESHOLD = 20       # Degrees for abnormal head pose
BLINK_FRAMES = 3               # Max frames for normal blink
DROWSINESS_FRAMES = 18         # ~0.6 seconds at 30fps (increased for accuracy)
YAWN_FRAMES = 20               # Consecutive frames for yawn detection
PERCLOS_WINDOW = 180           # 6-second rolling window for PERCLOS

# Alert settings
SOUND_FILE = '/home/abhijit/Downloads/mixkit-vintage-warning-alarm-990.wav'
LOG_FILE = 'drowsiness_log.txt'
USB_PORT = 'COM3'              # Update to your USB relay port (Windows: COM3, Linux: /dev/ttyUSB0)
USB_BAUD_RATE = 9600

# ==================== USB RELAY CONTROLLER ====================
class USBRelayController:
    """Controls USB relay module for alert light"""
    def __init__(self, port=USB_PORT, baud_rate=USB_BAUD_RATE):
        self.relay_active = False
        self.serial_conn = None
        if USE_USB_RELAY:
            try:
                import serial
                self.serial_conn = serial.Serial(port, baud_rate, timeout=1)
                time.sleep(2)  # Wait for connection
                print(f"USB Relay connected on {port}")
            except Exception as e:
                print(f"USB Relay connection failed: {e}")
                self.serial_conn = None
    
    def turn_on(self):
        """Turn on relay (light on)"""
        if self.serial_conn and not self.relay_active:
            try:
                # Command format varies by relay model
                # Common: 'A01' for channel 1 ON (update for your relay)
                self.serial_conn.write(b'A01')
                self.relay_active = True
                print("[RELAY] Light ON")
            except Exception as e:
                print(f"Relay ON error: {e}")
    
    def turn_off(self):
        """Turn off relay (light off)"""
        if self.serial_conn and self.relay_active:
            try:
                # Command: 'A00' for channel 1 OFF (update for your relay)
                self.serial_conn.write(b'A00')
                self.relay_active = False
                print("[RELAY] Light OFF")
            except Exception as e:
                print(f"Relay OFF error: {e}")
    
    def close(self):
        """Cleanup connection"""
        if self.serial_conn:
            self.turn_off()
            self.serial_conn.close()

# ==================== SOUND ALERT CONTROLLER ====================
class SoundAlertController:
    """Manages sound alerts with proper start/stop control"""
    def __init__(self, sound_file=SOUND_FILE):
        self.sound_file = sound_file
        self.is_playing = False
        self.stop_flag = threading.Event()
        self.play_thread = None
    
    def play(self):
        """Start playing alert sound in loop"""
        if not self.is_playing:
            self.is_playing = True
            self.stop_flag.clear()
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()
            print("[SOUND] Alert started")
    
    def stop(self):
        """Stop playing alert sound"""
        if self.is_playing:
            self.is_playing = False
            self.stop_flag.set()
            if platform.system() == "Windows":
                # Stop Windows sound
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except:
                    pass
            print("[SOUND] Alert stopped")
    
    def _play_loop(self):
        """Internal loop to repeat sound until stopped"""
        while not self.stop_flag.is_set():
            try:
                if os.path.exists(self.sound_file):
                    if platform.system() == "Windows":
                        winsound.PlaySound(self.sound_file, winsound.SND_FILENAME)
                    else:
                        subprocess.run(['aplay', self.sound_file], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
                else:
                    # Fallback beep
                    if platform.system() == "Windows":
                        winsound.Beep(2500, 500)
                time.sleep(0.1)  # Short pause between loops
            except Exception as e:
                print(f"Sound playback error: {e}")
                break

# ==================== CALCULATION FUNCTIONS ====================
def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
    A = np.linalg.norm(mouth[2] - mouth[10])  # Vertical distance 1
    B = np.linalg.norm(mouth[4] - mouth[8])   # Vertical distance 2
    C = np.linalg.norm(mouth[0] - mouth[6])   # Horizontal distance
    mar = (A + B) / (2.0 * C + 1e-6)
    return mar

def calculate_head_pose(landmarks, frame_shape):
    """Calculate head pose angles (pitch, yaw, roll)"""
    h, w = frame_shape[:2]
    
    # 3D model points (generic human head)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # Camera internals
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4,1))
    
    # 2D image points from landmarks
    image_points = np.array(landmarks, dtype=np.float64)
    
    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    
    # Convert rotation vector to angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch, yaw, roll = euler_angles.flatten()[:3]
    return pitch, yaw, roll

def calculate_perclos(eye_states, window_size=PERCLOS_WINDOW):
    """Calculate PERCLOS (Percentage of Eye Closure)"""
    if len(eye_states) < window_size:
        return 0.0
    recent_states = list(eye_states)[-window_size:]
    closed_count = sum(1 for state in recent_states if state)
    perclos = (closed_count / window_size) * 100
    return perclos

# ==================== MAIN DETECTION SYSTEM ====================
def show_alert_popup():
    """Show warning popup"""
    if USE_TK:
        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showwarning("⚠️ DROWSINESS ALERT", 
                                 "WAKE UP! You are showing signs of drowsiness!")
        except:
            print("DROWSINESS ALERT!")
        finally:
            root.destroy()

def main():
    # Initialize controllers
    sound_controller = SoundAlertController()
    relay_controller = USBRelayController()
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not video_capture.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    # Initialize face detection
    face_mesh = None
    if USE_MEDIAPIPE:
        try:
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("Using MediaPipe for detection")
        except Exception as e:
            print(f"MediaPipe init failed: {e}")
    
    # State tracking variables
    closed_eyes_frame_count = 0
    yawn_frame_count = 0
    closed_start_time = None
    alert_active = False
    eye_state_history = deque(maxlen=PERCLOS_WINDOW)
    
    # Statistics
    total_drowsy_events = 0
    total_yawns = 0
    
    print("\n=== Driver Drowsiness Detection System Active ===")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Warning: Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Default state
            status = "Monitoring..."
            color = (0, 255, 255)  # Yellow
            ear_value = 0.0
            mar_value = 0.0
            head_tilt = 0.0
            perclos = 0.0
            
            # Detection using MediaPipe
            if face_mesh is not None:
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Extract landmark coordinates
                    landmarks_coords = []
                    for lm in face_landmarks.landmark:
                        landmarks_coords.append([lm.x * w, lm.y * h])
                    landmarks_np = np.array(landmarks_coords)
                    
                    # ===== EYE ASPECT RATIO (EAR) =====
                    # Left eye indices: 33, 160, 158, 133, 153, 144
                    left_eye = landmarks_np[[33, 160, 158, 133, 153, 144]]
                    # Right eye indices: 263, 387, 385, 362, 380, 373
                    right_eye = landmarks_np[[263, 387, 385, 362, 380, 373]]
                    
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear_value = (left_ear + right_ear) / 2.0
                    
                    # ===== MOUTH ASPECT RATIO (MAR) =====
                    # Mouth indices: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308
                    mouth_points = landmarks_np[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]]
                    mar_value = mouth_aspect_ratio(mouth_points)
                    
                    # ===== HEAD POSE ESTIMATION =====
                    try:
                        # Key points for pose: nose, chin, left eye, right eye, left mouth, right mouth
                        pose_points = [
                            landmarks_coords[1],    # Nose tip
                            landmarks_coords[152],  # Chin
                            landmarks_coords[33],   # Left eye corner
                            landmarks_coords[263],  # Right eye corner
                            landmarks_coords[61],   # Left mouth corner
                            landmarks_coords[291]   # Right mouth corner
                        ]
                        pitch, yaw, roll = calculate_head_pose(pose_points, frame.shape)
                        head_tilt = abs(pitch)
                    except:
                        head_tilt = 0.0
                    
                    # ===== DROWSINESS DETECTION LOGIC =====
                    eyes_closed = ear_value < EAR_THRESHOLD
                    is_yawning = mar_value > MAR_THRESHOLD
                    head_nodding = head_tilt > HEAD_TILT_THRESHOLD
                    
                    # Track eye state for PERCLOS
                    eye_state_history.append(eyes_closed)
                    perclos = calculate_perclos(eye_state_history)
                    
                    # Eyes closed detection
                    if eyes_closed:
                        if closed_start_time is None:
                            closed_start_time = time.time()
                        closed_eyes_frame_count += 1
                        
                        # Check if it's actual drowsiness (not just blinking)
                        if closed_eyes_frame_count >= DROWSINESS_FRAMES:
                            status = "⚠️ DROWSINESS DETECTED!"
                            color = (0, 0, 255)  # Red
                            
                            if not alert_active:
                                alert_active = True
                                total_drowsy_events += 1
                                print(f"\n[ALERT] Drowsiness detected! EAR={ear_value:.3f}, PERCLOS={perclos:.1f}%")
                                
                                # Trigger all alerts
                                sound_controller.play()
                                relay_controller.turn_on()
                                threading.Thread(target=show_alert_popup, daemon=True).start()
                        else:
                            status = f"Eyes Closing... ({closed_eyes_frame_count}/{DROWSINESS_FRAMES})"
                            color = (0, 165, 255)  # Orange
                    
                    else:
                        # Eyes are open - STOP ALERTS
                        if closed_start_time is not None:
                            duration = time.time() - closed_start_time
                            if duration > 0.5:  # Only log if eyes were closed >0.5s
                                ts = datetime.now().isoformat(sep=' ', timespec='seconds')
                                entry = f"{ts} - Eyes closed for {duration:.2f}s, PERCLOS: {perclos:.1f}%\n"
                                try:
                                    with open(LOG_FILE, 'a') as f:
                                        f.write(entry)
                                    print(f"[LOG] {entry.strip()}")
                                except Exception as e:
                                    print(f"Log error: {e}")
                            
                            closed_start_time = None
                            closed_eyes_frame_count = 0
                        
                        # Stop all alerts when eyes reopen
                        if alert_active:
                            alert_active = False
                            sound_controller.stop()
                            relay_controller.turn_off()
                            print("[SYSTEM] Driver alert - alerts stopped")
                        
                        status = "Awake"
                        color = (0, 255, 0)  # Green
                    
                    # ===== YAWN DETECTION =====
                    if is_yawning:
                        yawn_frame_count += 1
                        if yawn_frame_count >= YAWN_FRAMES:
                            status += " + YAWNING"
                            color = (0, 140, 255)  # Dark orange
                            if yawn_frame_count == YAWN_FRAMES:
                                total_yawns += 1
                                print(f"[YAWN] Detected! MAR={mar_value:.3f}")
                    else:
                        yawn_frame_count = 0
                    
                    # ===== HEAD NODDING DETECTION =====
                    if head_nodding and not eyes_closed:
                        status += " + HEAD NODDING"
                        print(f"[HEAD] Abnormal pose detected: {head_tilt:.1f}°")
            
            # ===== DISPLAY OVERLAY =====
            # Status banner
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Metrics panel
            metrics_y = 80
            cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, metrics_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar_value:.3f}", (10, metrics_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (10, metrics_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Head: {head_tilt:.1f}deg", (10, metrics_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Statistics
            cv2.putText(frame, f"Drowsy Events: {total_drowsy_events}", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Yawns: {total_yawns}", (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Alert indicator
            if alert_active:
                cv2.rectangle(frame, (w-150, 10), (w-10, 50), (0, 0, 255), -1)
                cv2.putText(frame, "ALERT!", (w-135, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Driver Drowsiness Detection System", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print('\n[EXIT] Interrupted by user')
    
    finally:
        # Cleanup
        print("\n[CLEANUP] Shutting down...")
        sound_controller.stop()
        relay_controller.close()
        video_capture.release()
        cv2.destroyAllWindows()
        if face_mesh:
            face_mesh.close()
        print("[EXIT] System stopped successfully")
        print(f"\nSession Stats - Drowsy Events: {total_drowsy_events}, Yawns: {total_yawns}")

if __name__ == "__main__":
    main()
