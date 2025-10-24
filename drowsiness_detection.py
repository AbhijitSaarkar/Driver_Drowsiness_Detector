import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
from collections import deque
import platform
import traceback

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

# WhatsApp Integration
try:
    from twilio.rest import Client
    USE_WHATSAPP = True
    print("✅ Twilio library loaded successfully")
except ImportError:
    USE_WHATSAPP = False
    print("❌ Twilio not installed. Install with: pip install twilio")

# ==================== CONFIGURATION ====================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection thresholds
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.5
HEAD_TILT_THRESHOLD = 20
BLINK_FRAMES = 3
DROWSINESS_FRAMES = 18  # Change to 5 for easier testing
YAWN_FRAMES = 20
PERCLOS_WINDOW = 180

# Alert settings
SOUND_FILE = '/home/abhijit/Downloads/mixkit-vintage-warning-alarm-990.wav'
LOG_FILE = 'drowsiness_log.txt'
USB_PORT = 'COM3'
USB_BAUD_RATE = 9600

# WhatsApp Configuration
import os

# It's safer to load Twilio credentials from environment variables so secrets
# are not stored in the repository. Set these in your shell or in a `.env`
# file loaded by your runtime environment.
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'REDACTED_TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'REDACTED_TWILIO_AUTH')
TWILIO_WHATSAPP_FROM = 'whatsapp:+14155238886'
DRIVER_WHATSAPP_NUMBER = 'whatsapp:+917001511540'
WHATSAPP_COOLDOWN_SECONDS = 120

# Debug mode
DEBUG_MODE = True
# Force attempts to send WhatsApp even if Twilio python package isn't installed.
# If True the code will try the REST API directly using requests or urllib.
FORCE_WHATSAPP_REST = True


# ==================== WHATSAPP ALERT SYSTEM ====================
class WhatsAppAlertSystem:
    """Sends WhatsApp alerts via Twilio when driver is sleeping"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING WHATSAPP ALERT SYSTEM")
        print("="*70)

        # enabled indicates whether we should attempt sending.
        # Allow sending when either Twilio SDK is available or REST fallback is allowed.
        self.enabled = USE_WHATSAPP or FORCE_WHATSAPP_REST
        self.client = None
        self.last_alert_time = None
        self.cooldown_seconds = WHATSAPP_COOLDOWN_SECONDS

        if not self.enabled:
            print("❌ Twilio library not available and REST fallback disabled")
            print("   Install with: pip install twilio or enable FORCE_WHATSAPP_REST")
            print("="*70 + "\n")
            return

        print(f"📱 Account SID: {TWILIO_ACCOUNT_SID[:20]}...")
        print(f"📞 From: {TWILIO_WHATSAPP_FROM}")
        print(f"📞 To: {DRIVER_WHATSAPP_NUMBER}")

        try:
            print("\n🔄 Creating Twilio client...")
            if USE_WHATSAPP:
                # prefer SDK when installed
                self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                print("✅ Twilio client created successfully!")
            else:
                self.client = None
                print("⚠️ Twilio SDK not available; will use REST fallback if needed.")

            # NOTE: Don't run a blocking startup test here. In some environments
            # the interactive test (which asks for ENTER) can pause the program
            # or a failed sandbox join at startup will permanently disable alerts.
            # Instead, we initialize the client and allow sends to be attempted
            # at alert time. To run a manual test, call send_test_message().
            print("\nℹ️ Twilio client initialized. Skipping automatic startup test.")

        except Exception as e:
            print(f"\n❌ FAILED to initialize Twilio!")
            print(f"Error: {e}")
            traceback.print_exc()
            # don't disable entirely; keep enabled if REST fallback is allowed
            if not FORCE_WHATSAPP_REST:
                self.enabled = False
            print("\n🔧 Check your Account SID and Auth Token at:")
            print("   https://console.twilio.com")
            print("="*70 + "\n")

    def _rest_send_message(self, from_, to, body):
        """Send message using Twilio REST API directly (fallback).
        Returns response-like dict or raises exception on failure.
        """
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            'From': from_,
            'To': to,
            'Body': body
        }
        # Try requests if available
        try:
            import requests
            if DEBUG_MODE:
                print("   ℹ️ Using requests library for REST POST to Twilio")
            resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
            if resp.status_code >= 200 and resp.status_code < 300:
                try:
                    return resp.json()
                except Exception:
                    return {'status_code': resp.status_code, 'text': resp.text}
            else:
                raise Exception(f"Twilio REST error: {resp.status_code} {resp.text}")
        except Exception as e:
            # Fallback to urllib
            if DEBUG_MODE:
                print(f"   ⚠️ requests failed or unavailable for REST send: {e}")
            try:
                from urllib import parse, request
                data_enc = parse.urlencode(data).encode()
                req = request.Request(url, data=data_enc)
                # Add basic auth header
                import base64
                credentials = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}"
                b64 = base64.b64encode(credentials.encode()).decode()
                req.add_header('Authorization', f'Basic {b64}')
                with request.urlopen(req, timeout=10) as res:
                    text = res.read().decode()
                    try:
                        import json
                        return json.loads(text)
                    except Exception:
                        return {'status_code': res.getcode(), 'text': text}
            except Exception as e2:
                raise Exception(f"REST fallback failed: {e2}")
    
    def send_test_message(self):
        """Send test message on startup"""
        if not self.enabled or not self.client:
            return False
        
        try:
            print("   📤 Sending test message...")
            
            message = self.client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body="✅ TEST MESSAGE\n\nDriver Drowsiness Detection System is starting up!\n\nIf you see this, WhatsApp alerts are working! 🎉",
                to=DRIVER_WHATSAPP_NUMBER
            )
            
            print(f"\n   ✅✅✅ TEST MESSAGE SENT! ✅✅✅")
            print(f"   Message SID: {message.sid}")
            print(f"   Status: {message.status}")
            print(f"\n   📱 CHECK YOUR WHATSAPP NOW!")
            print(f"   Number: {DRIVER_WHATSAPP_NUMBER}")
            print("="*70 + "\n")
            
            # Wait for user confirmation
            print("⏸️  Did you receive the test message on WhatsApp?")
            print("   If YES: Press ENTER to continue")
            print("   If NO: Check the error message above\n")
            
            try:
                input("Press ENTER when ready...")
            except:
                print("Auto-continuing in 5 seconds...")
                time.sleep(5)
            
            return True
            
        except Exception as e:
            print(f"\n   ❌❌❌ TEST MESSAGE FAILED! ❌❌❌")
            print(f"   Error: {e}\n")
            error_str = str(e).lower()
            if "21608" in error_str or "not a valid" in error_str:
                print("   🚫 PROBLEM: YOU HAVEN'T JOINED THE SANDBOX!")
                print("   SOLUTION: From your phone's WhatsApp, message the Twilio sandbox number and send the join code shown in your Twilio Console.")
            elif "authenticate" in error_str or "20003" in error_str:
                print("   🔑 PROBLEM: WRONG CREDENTIALS! Check Account SID/Auth Token in your config.")
            else:
                print("   ❓ UNKNOWN ERROR: Check traceback below for details.")
                traceback.print_exc()

            # Do NOT permanently disable WhatsApp here; keep enabled flag as-is so
            # the system can attempt sends later and the operator can re-run
            # send_test_message manually after fixing credentials/sandbox.
            print("\n" + "="*70 + "\n")
            return False
    
    def send_drowsiness_alert(self, ear_value, perclos, total_events=None, total_yawns=None, session_duration=None, force=False):
        """Send WhatsApp alert when driver is sleeping. Includes optional session stats."""
        
        if DEBUG_MODE:
            print(f"\n🚨 WHATSAPP ALERT TRIGGERED!")
            print(f"   EAR: {ear_value:.3f}")
            print(f"   PERCLOS: {perclos:.1f}%")
        
        if not self.enabled:
            if DEBUG_MODE:
                print("   ❌ WhatsApp not enabled - skipping")
            return False
        
        # If the Twilio SDK client isn't initialized that's OK —
        # we will attempt the REST fallback when available (FORCE_WHATSAPP_REST).
        # Do NOT return early here, otherwise REST sends are never attempted.
        if not self.client and DEBUG_MODE:
            print("   ⚠️ Twilio client not initialized; will attempt REST fallback if configured")
        
        # Check cooldown (unless forced)
        current_time = datetime.now()
        if self.last_alert_time and not force:
            elapsed = (current_time - self.last_alert_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                if DEBUG_MODE:
                    print(f"   ⏸️ Cooldown: Last alert {int(elapsed)}s ago (need {self.cooldown_seconds}s)")
                return False
        elif force and DEBUG_MODE:
            print("   ⚡ Force send requested — skipping cooldown")
        
        try:
            stats_text = ""
            if total_events is not None or total_yawns is not None or session_duration is not None:
                stats_text = "\n\n📊 Session stats:\n"
                if total_events is not None:
                    stats_text += f"- Drowsy events: {total_events}\n"
                if total_yawns is not None:
                    stats_text += f"- Yawns: {total_yawns}\n"
                if session_duration is not None:
                    stats_text += f"- Session duration: {session_duration}\n"

            alert_message = f"""🚨🚨🚨 *DRIVER IS SLEEPING* 🚨🚨🚨

⚠️ *CRITICAL ALERT* ⚠️

🕐 *Time:* {current_time.strftime('%I:%M:%S %p')}
👁️ *Eye Status (EAR):* {ear_value:.3f}
📊 *Drowsiness Level (PERCLOS):* {perclos:.1f}%{stats_text}

⛔ *IMMEDIATE ACTION REQUIRED!*
The driver is showing signs of sleep. Check on them NOW and ensure they pull over safely.

🚗 Stay Safe! 🚗"""

            print(f"   📤 Sending WhatsApp message...")
            # Prefer SDK if available, otherwise use REST fallback
            if self.client:
                message = self.client.messages.create(
                    from_=TWILIO_WHATSAPP_FROM,
                    body=alert_message,
                    to=DRIVER_WHATSAPP_NUMBER
                )
                if DEBUG_MODE:
                    print("   ✅ WHATSAPP ALERT SENT via SDK")
                    try:
                        print(f"   Message SID: {message.sid}")
                        print(f"   Status: {getattr(message, 'status', 'unknown')}")
                    except Exception:
                        pass
            else:
                # REST fallback
                resp = self._rest_send_message(TWILIO_WHATSAPP_FROM, DRIVER_WHATSAPP_NUMBER, alert_message)
                if DEBUG_MODE:
                    print("   ✅ WHATSAPP ALERT SENT via REST")
                    try:
                        if isinstance(resp, dict) and 'sid' in resp:
                            print(f"   Message SID: {resp.get('sid')}")
                    except Exception:
                        pass

            # mark last alert time on success
            self.last_alert_time = current_time

            # write a short entry into the local log so operator can see send attempts
            try:
                with open(LOG_FILE, 'a') as _lf:
                    _lf.write(f"{datetime.now().isoformat(sep=' ', timespec='seconds')} - WhatsApp alert sent (EAR={ear_value:.3f}, PERCLOS={perclos:.1f}%, Events={total_events}, Yawns={total_yawns})\n")
            except Exception:
                pass

            print("\n")
            return True
            
        except Exception as e:
            print(f"   ❌ WhatsApp alert FAILED!")
            print(f"   Error: {e}\n")
            traceback.print_exc()
            return False
    
    def send_session_summary(self, total_events, total_yawns, session_duration):
        """Send summary when system shuts down"""
        if not self.enabled or not self.client:
            return
        
        try:
            summary = f"""📊 *Session Summary*

Total Drowsiness Events: {total_events}
Total Yawns Detected: {total_yawns}
Session Duration: {session_duration}

System has been shut down.
Stay safe! 🚗"""
            
            message = self.client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body=summary,
                to=DRIVER_WHATSAPP_NUMBER
            )
            print(f"📱 Session summary sent: {message.sid}")
        except Exception as e:
            print(f"❌ Summary failed: {e}")


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
                time.sleep(2)
                print(f"✅ USB Relay connected on {port}")
            except Exception as e:
                print(f"⚠️ USB Relay failed: {e}")
                self.serial_conn = None
    
    def turn_on(self):
        if self.serial_conn and not self.relay_active:
            try:
                self.serial_conn.write(b'A01')
                self.relay_active = True
                print("[RELAY] Light ON")
            except Exception as e:
                print(f"Relay ON error: {e}")
    
    def turn_off(self):
        if self.serial_conn and self.relay_active:
            try:
                self.serial_conn.write(b'A00')
                self.relay_active = False
                print("[RELAY] Light OFF")
            except Exception as e:
                print(f"Relay OFF error: {e}")
    
    def close(self):
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
        if not self.is_playing:
            self.is_playing = True
            self.stop_flag.clear()
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()
            print("[SOUND] Alert started")
    
    def stop(self):
        if self.is_playing:
            self.is_playing = False
            self.stop_flag.set()
            if platform.system() == "Windows":
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except:
                    pass
            print("[SOUND] Alert stopped")
    
    def _play_loop(self):
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
                    if platform.system() == "Windows":
                        winsound.Beep(2500, 500)
                time.sleep(0.1)
            except Exception as e:
                print(f"Sound error: {e}")
                break


# ==================== CALCULATION FUNCTIONS ====================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C + 1e-6)
    return mar

def calculate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    image_points = np.array(landmarks, dtype=np.float64)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()[:3]
    return pitch, yaw, roll

def calculate_perclos(eye_states, window_size=PERCLOS_WINDOW):
    if len(eye_states) < window_size:
        return 0.0
    recent_states = list(eye_states)[-window_size:]
    closed_count = sum(1 for state in recent_states if state)
    perclos = (closed_count / window_size) * 100
    return perclos


# ==================== MAIN DETECTION SYSTEM ====================
def show_alert_popup():
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
    session_start = datetime.now()
    
    print("\n" + "="*70)
    print("DRIVER DROWSINESS DETECTION SYSTEM")
    print("="*70 + "\n")
    
    # Initialize controllers
    print("Initializing system components...\n")
    sound_controller = SoundAlertController()
    relay_controller = USBRelayController()
    whatsapp_alert = WhatsAppAlertSystem()
    
    # Initialize video capture
    print("\nInitializing camera...")
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not video_capture.isOpened():
        print("❌ ERROR: Cannot open camera")
        return
    
    print("✅ Camera initialized")
    
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
            print("✅ MediaPipe face detection initialized")
        except Exception as e:
            print(f"❌ MediaPipe failed: {e}")
    
    # State tracking
    closed_eyes_frame_count = 0
    yawn_frame_count = 0
    closed_start_time = None
    alert_active = False
    eye_state_history = deque(maxlen=PERCLOS_WINDOW)
    
    total_drowsy_events = 0
    total_yawns = 0
    
    print("\n" + "="*70)
    print("SYSTEM ACTIVE - MONITORING STARTED")
    print("="*70)
    print("📹 Camera: Active")
    print("🎵 Sound Alerts: Active")
    print(f"📱 WhatsApp Alerts: {'✅ Active' if whatsapp_alert.enabled else '❌ Inactive'}")
    print("\n💡 TIP: Close your eyes for 1-2 seconds to test")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Warning: Frame capture failed")
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            status = "Monitoring..."
            color = (0, 255, 255)
            ear_value = 0.0
            mar_value = 0.0
            head_tilt = 0.0
            perclos = 0.0
            
            if face_mesh is not None:
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    landmarks_coords = []
                    for lm in face_landmarks.landmark:
                        landmarks_coords.append([lm.x * w, lm.y * h])
                    landmarks_np = np.array(landmarks_coords)
                    
                    # EYE ASPECT RATIO
                    left_eye = landmarks_np[[33, 160, 158, 133, 153, 144]]
                    right_eye = landmarks_np[[263, 387, 385, 362, 380, 373]]
                    
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear_value = (left_ear + right_ear) / 2.0
                    
                    # MOUTH ASPECT RATIO
                    mouth_points = landmarks_np[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]]
                    mar_value = mouth_aspect_ratio(mouth_points)
                    
                    # HEAD POSE
                    try:
                        pose_points = [
                            landmarks_coords[1],
                            landmarks_coords[152],
                            landmarks_coords[33],
                            landmarks_coords[263],
                            landmarks_coords[61],
                            landmarks_coords[291]
                        ]
                        pitch, yaw, roll = calculate_head_pose(pose_points, frame.shape)
                        head_tilt = abs(pitch)
                    except:
                        head_tilt = 0.0
                    
                    # DROWSINESS DETECTION
                    eyes_closed = ear_value < EAR_THRESHOLD
                    is_yawning = mar_value > MAR_THRESHOLD
                    head_nodding = head_tilt > HEAD_TILT_THRESHOLD
                    
                    eye_state_history.append(eyes_closed)
                    perclos = calculate_perclos(eye_state_history)
                    
                    if eyes_closed:
                        if closed_start_time is None:
                            closed_start_time = time.time()
                        closed_eyes_frame_count += 1
                        
                        if DEBUG_MODE:
                            print(f"👁️ Eyes closed: {closed_eyes_frame_count}/{DROWSINESS_FRAMES} frames | EAR: {ear_value:.3f}")
                        
                        if closed_eyes_frame_count >= DROWSINESS_FRAMES:
                            status = "⚠️ DROWSINESS DETECTED!"
                            color = (0, 0, 255)
                            
                            if not alert_active:
                                alert_active = True
                                total_drowsy_events += 1
                                
                                print(f"\n{'='*70}")
                                print(f"🚨 DROWSINESS ALERT #{total_drowsy_events}")
                                print(f"{'='*70}")
                                print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
                                print(f"EAR: {ear_value:.3f}")
                                print(f"PERCLOS: {perclos:.1f}%")
                                print(f"{'='*70}\n")
                                
                                # Trigger all alerts
                                sound_controller.play()
                                relay_controller.turn_on()
                                threading.Thread(target=show_alert_popup, daemon=True).start()
                                
                                # Send WhatsApp alert
                                # compute session duration so far as H:MM:SS
                                session_duration_str = str(datetime.now() - session_start).split('.')[0]
                                # Force send so cooldown is ignored — this makes sure every
                                # time the alarm rings an alert message is sent.
                                threading.Thread(
                                    target=whatsapp_alert.send_drowsiness_alert,
                                    args=(ear_value, perclos, total_drowsy_events, total_yawns, session_duration_str, True),
                                    daemon=True
                                ).start()
                        else:
                            status = f"Eyes Closing... ({closed_eyes_frame_count}/{DROWSINESS_FRAMES})"
                            color = (0, 165, 255)
                    
                    else:
                        # Eyes open - stop alerts
                        if closed_start_time is not None:
                            duration = time.time() - closed_start_time
                            if duration > 0.5:
                                ts = datetime.now().isoformat(sep=' ', timespec='seconds')
                                entry = f"{ts} - Eyes closed for {duration:.2f}s, PERCLOS: {perclos:.1f}%\n"
                                try:
                                    with open(LOG_FILE, 'a') as f:
                                        f.write(entry)
                                    if DEBUG_MODE:
                                        print(f"[LOG] {entry.strip()}")
                                except Exception as e:
                                    print(f"Log error: {e}")
                            
                            closed_start_time = None
                            closed_eyes_frame_count = 0
                        
                        if alert_active:
                            alert_active = False
                            sound_controller.stop()
                            relay_controller.turn_off()
                            print("✅ Driver awake - All alerts stopped\n")
                        
                        status = "Awake"
                        color = (0, 255, 0)
                    
                    # YAWN DETECTION
                    if is_yawning:
                        yawn_frame_count += 1
                        if yawn_frame_count >= YAWN_FRAMES:
                            status += " + YAWNING"
                            color = (0, 140, 255)
                            if yawn_frame_count == YAWN_FRAMES:
                                total_yawns += 1
                                print(f"🥱 Yawn #{total_yawns} detected | MAR: {mar_value:.3f}")
                    else:
                        yawn_frame_count = 0
                    
                    # HEAD NODDING
                    if head_nodding and not eyes_closed:
                        status += " + HEAD NODDING"
                        if DEBUG_MODE:
                            print(f"[HEAD] Abnormal pose: {head_tilt:.1f}°")
            
            # DISPLAY
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            metrics_y = 80
            cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, metrics_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar_value:.3f}", (10, metrics_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (10, metrics_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Head: {head_tilt:.1f}deg", (10, metrics_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Drowsy Events: {total_drowsy_events}", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Yawns: {total_yawns}", (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if alert_active:
                cv2.rectangle(frame, (w-150, 10), (w-10, 50), (0, 0, 255), -1)
                cv2.putText(frame, "ALERT!", (w-135, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Driver Drowsiness Detection System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print('\n[EXIT] Interrupted by user')
    
    finally:
        print("\n[CLEANUP] Shutting down...")
        sound_controller.stop()
        relay_controller.close()
        
        session_end = datetime.now()
        session_duration = str(session_end - session_start).split('.')[0]
        
        whatsapp_alert.send_session_summary(total_drowsy_events, total_yawns, session_duration)
        
        video_capture.release()
        cv2.destroyAllWindows()
        if face_mesh:
            face_mesh.close()
        
        print("[EXIT] System stopped successfully")
        print(f"\n📊 Final Stats:")
        print(f"   Drowsy Events: {total_drowsy_events}")
        print(f"   Yawns: {total_yawns}")
        print(f"   Duration: {session_duration}")


if __name__ == "__main__":
    main()

