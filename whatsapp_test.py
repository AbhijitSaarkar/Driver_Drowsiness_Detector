import time
import traceback
from datetime import datetime

# Copy your Twilio config from the main script
import os

# Load Twilio credentials from environment (recommended)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'REDACTED_TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'REDACTED_TWILIO_AUTH')
TWILIO_WHATSAPP_FROM = 'whatsapp:+14155238886'
DRIVER_WHATSAPP_NUMBER = 'whatsapp:+917001511540'

DEBUG_MODE = True


def rest_send_message(from_, to, body):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {'From': from_, 'To': to, 'Body': body}
    # Try requests first (preferred)
    try:
        import requests
        if DEBUG_MODE:
            print('Using requests for REST POST')
        resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=15)
        if resp.status_code >= 200 and resp.status_code < 300:
            try:
                return resp.json()
            except Exception:
                return {'status_code': resp.status_code, 'text': resp.text}
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        if DEBUG_MODE:
            print('requests failed or not present, falling back to urllib:', e)
        try:
            from urllib import parse, request
            import base64
            import json
            data_enc = parse.urlencode(data).encode()
            req = request.Request(url, data=data_enc)
            credentials = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}"
            b64 = base64.b64encode(credentials.encode()).decode()
            req.add_header('Authorization', f'Basic {b64}')
            with request.urlopen(req, timeout=15) as res:
                text = res.read().decode()
                try:
                    return json.loads(text)
                except Exception:
                    return {'status_code': res.getcode(), 'text': text}
        except Exception as e2:
            raise Exception(f"urllib fallback failed: {e2}")


if __name__ == '__main__':
    try:
        body = f"✅ TEST MESSAGE\n\nTime: {datetime.now().isoformat(sep=' ', timespec='seconds')}\nThis is a test from whatsapp_test.py"
        print('Sending test WhatsApp message to', DRIVER_WHATSAPP_NUMBER)
        resp = rest_send_message(TWILIO_WHATSAPP_FROM, DRIVER_WHATSAPP_NUMBER, body)
        print('Send response:\n', resp)
    except Exception as e:
        print('Send failed: ', e)
        traceback.print_exc()
