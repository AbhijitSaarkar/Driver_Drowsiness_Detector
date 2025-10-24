"""
Lightweight Twilio WhatsApp client used by the main app and test scripts.
Provides SDK + REST fallback and clear logging.
"""
import traceback
from datetime import datetime

# Configuration - keep in sync with the main script or centralize later
import os

# Load Twilio credentials from environment to avoid committing secrets.
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'REDACTED_TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'REDACTED_TWILIO_AUTH')
TWILIO_WHATSAPP_FROM = 'whatsapp:+14155238886'
DRIVER_WHATSAPP_NUMBER = 'whatsapp:+917001511540'
WHATSAPP_COOLDOWN_SECONDS = 120

DEBUG_MODE = True
FORCE_WHATSAPP_REST = True


class WhatsAppAlertSystem:
    """Sends WhatsApp alerts via Twilio when driver is sleeping.

    Features:
    - Use Twilio Python SDK when available.
    - Fallback to Twilio REST API using requests, then urllib.
    - Non-blocking initialization (no interactive tests run automatically).
    - Simple local logging to help debugging.
    """

    def __init__(self, account_sid=None, auth_token=None, from_number=None, to_number=None, cooldown=None):
        # allow overrides
        self.account_sid = account_sid or TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or TWILIO_AUTH_TOKEN
        self.from_number = from_number or TWILIO_WHATSAPP_FROM
        self.to_number = to_number or DRIVER_WHATSAPP_NUMBER
        self.cooldown_seconds = cooldown or WHATSAPP_COOLDOWN_SECONDS

        # Try to import Twilio SDK
        try:
            from twilio.rest import Client
            self.Client = Client
            self.use_sdk = True
            if DEBUG_MODE:
                print("✅ Twilio SDK available")
        except Exception:
            self.Client = None
            self.use_sdk = False
            if DEBUG_MODE:
                print("⚠️ Twilio SDK not available; will use REST fallback")

        # enabled if SDK present or REST fallback allowed
        self.enabled = self.use_sdk or FORCE_WHATSAPP_REST
        self.client = None
        self.last_alert_time = None

        if self.use_sdk:
            try:
                self.client = self.Client(self.account_sid, self.auth_token)
                if DEBUG_MODE:
                    print("🔄 Twilio client initialized (SDK)")
            except Exception as e:
                if DEBUG_MODE:
                    print("❌ Failed to create Twilio SDK client:", e)
                    traceback.print_exc()
                # allow REST fallback
                self.client = None
                self.use_sdk = False

        if DEBUG_MODE:
            print(f"📞 From: {self.from_number}  → To: {self.to_number}")
            print(f"📱 WhatsApp enabled: {'yes' if self.enabled else 'no'}")

    def _rest_send_message(self, from_, to, body):
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {'From': from_, 'To': to, 'Body': body}
        # try requests
        try:
            import requests
            if DEBUG_MODE:
                print("   ℹ️ Using requests library for REST POST to Twilio")
            resp = requests.post(url, data=data, auth=(self.account_sid, self.auth_token), timeout=10)
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except Exception:
                    return {'status_code': resp.status_code, 'text': resp.text}
            else:
                raise Exception(f"Twilio REST error: {resp.status_code} {resp.text}")
        except Exception as e:
            if DEBUG_MODE:
                print(f"   ⚠️ requests failed/unavailable for REST send: {e}")
            # fallback to urllib
            try:
                from urllib import parse, request
                import base64, json
                data_enc = parse.urlencode(data).encode()
                req = request.Request(url, data=data_enc)
                credentials = f"{self.account_sid}:{self.auth_token}"
                b64 = base64.b64encode(credentials.encode()).decode()
                req.add_header('Authorization', f'Basic {b64}')
                with request.urlopen(req, timeout=10) as res:
                    text = res.read().decode()
                    try:
                        return json.loads(text)
                    except Exception:
                        return {'status_code': res.getcode(), 'text': text}
            except Exception as e2:
                raise Exception(f"REST fallback failed: {e2}")

    def send_test_message(self):
        if not self.enabled:
            if DEBUG_MODE:
                print("   ❌ WhatsApp not enabled - skipping test send")
            return False
        try:
            body = f"✅ TEST MESSAGE\n\nTime: {datetime.now().isoformat(sep=' ', timespec='seconds')}\nThis is a test message."
            if self.client:
                message = self.client.messages.create(from_=self.from_number, body=body, to=self.to_number)
                if DEBUG_MODE:
                    print("   ✅ TEST MESSAGE SENT via SDK")
                    print("   sid:", getattr(message, 'sid', None))
                return True
            else:
                resp = self._rest_send_message(self.from_number, self.to_number, body)
                if DEBUG_MODE:
                    print("   ✅ TEST MESSAGE SENT via REST")
                    print(resp)
                return True
        except Exception as e:
            print("   ❌ TEST MESSAGE FAILED:", e)
            traceback.print_exc()
            return False
    def send_drowsiness_alert(self, ear_value, perclos, yawns=None):
        """Send an urgent drowsiness alert including optional yawns count."""
        if not self.enabled:
            if DEBUG_MODE:
                print("   ❌ WhatsApp not enabled - skipping")
            return False

        current_time = datetime.now()
        # cooldown check
        if self.last_alert_time:
            elapsed = (current_time - self.last_alert_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                if DEBUG_MODE:
                    print(f"   ⏸️ Cooldown: Last alert {int(elapsed)}s ago (need {self.cooldown_seconds}s)")
                return False

        try:
            yawns_text = f"\nYawns detected this session: {yawns}" if yawns is not None else ""
            alert_message = (
                f"🚨 DRIVER IS SLEEPING 🚨\n"
                f"Time: {current_time.strftime('%I:%M:%S %p')}\n"
                f"EAR: {ear_value:.3f}\n"
                f"PERCLOS: {perclos:.1f}%{yawns_text}"
            )

            if self.client:
                message = self.client.messages.create(from_=self.from_number, body=alert_message, to=self.to_number)
                if DEBUG_MODE:
                    print("   ✅ WHATSAPP ALERT SENT via SDK", getattr(message, 'sid', None))
            else:
                resp = self._rest_send_message(self.from_number, self.to_number, alert_message)
                if DEBUG_MODE:
                    print("   ✅ WHATSAPP ALERT SENT via REST", resp.get('sid') if isinstance(resp, dict) else resp)

            self.last_alert_time = current_time

            # append small log that includes yawns
            try:
                with open('drowsiness_log.txt', 'a') as f:
                    f.write(f"{datetime.now().isoformat(sep=' ', timespec='seconds')} - WhatsApp alert sent (EAR={ear_value:.3f}, PERCLOS={perclos:.1f}%, Yawns={yawns})\n")
            except Exception:
                pass

            return True
        except Exception as e:
            print("   ❌ WhatsApp alert FAILED:", e)
            traceback.print_exc()
            return False

    def send_session_summary(self, total_events, total_yawns, session_duration):
        """Send a session summary message when system shuts down."""
        if not self.enabled:
            return False
        try:
            summary = (
                f"� Session Summary\n\n"
                f"Total Drowsiness Events: {total_events}\n"
                f"Total Yawns Detected: {total_yawns}\n"
                f"Session Duration: {session_duration}\n\n"
                "System has been shut down. Stay safe! 🚗"
            )
            if self.client:
                message = self.client.messages.create(from_=self.from_number, body=summary, to=self.to_number)
                if DEBUG_MODE:
                    print("   ✅ Session summary sent via SDK", getattr(message, 'sid', None))
            else:
                resp = self._rest_send_message(self.from_number, self.to_number, summary)
                if DEBUG_MODE:
                    print("   ✅ Session summary sent via REST", resp.get('sid') if isinstance(resp, dict) else resp)
            return True
        except Exception as e:
            print("   ❌ Summary failed:", e)
            traceback.print_exc()
            return False
