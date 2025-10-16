# her_safe_ai_prototype.py
"""
HerSafe Prototype
Features:
 - Continuous microphone listening (or file) + transcription (SpeechRecognition)
 - Keyword-based distress detection + simple urgency heuristic
 - Chat/text monitor (scan list or file) for distress keywords
 - Alert function: prints and can send SMS via Twilio if configured

Author: Ramana M (prototype)
Dates: Start 2025-10-13 End 2025-10-16
"""

import time
import threading
import queue
import os
import math
from typing import List
import speech_recognition as sr

# Optional Twilio. If not installed or not configured, alerts will only print to console.
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# -------------------------
# CONFIGURATION
# -------------------------
CONFIG = {
    "keywords": ["help", "help me", "save me", "stop", "rape", "followed", "danger", "attack", "assault", "call", "police"],
    "urgent_indicators": ["now", "please", "urgent", "immediately", "right now"],
    "rms_threshold": 300,
    "keyword_match_threshold": 1,
    "twilio": {
        "account_sid": os.environ.get("TWILIO_ACCOUNT_SID", ""),
        "auth_token": os.environ.get("TWILIO_AUTH_TOKEN", ""),
        "from_number": os.environ.get("TWILIO_FROM_NUMBER", ""),
        "trusted_contact": os.environ.get("TRUSTED_CONTACT_NUMBER", ""),
    },
    "alert_cooldown": 60,
    "chat_poll_interval": 5,
}

_last_alert_time = 0
twilio_client = None

# -------------------------
# Utilities & Alerting
# -------------------------
def init_twilio():
    global twilio_client
    cfg = CONFIG["twilio"]
    if TWILIO_AVAILABLE and cfg["account_sid"] and cfg["auth_token"]:
        twilio_client = TwilioClient(cfg["account_sid"], cfg["auth_token"])
        return True
    return False

def send_alert_via_twilio(message: str):
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time < CONFIG["alert_cooldown"]:
        print("[Alert] Cooldown active - skipping SMS.")
        return False

    if not twilio_client:
        print("[Alert] Twilio not configured. Install twilio & set TWILIO_* env vars to enable SMS.")
        return False

    cfg = CONFIG["twilio"]
    try:
        twilio_client.messages.create(
            body=message,
            from_=cfg["from_number"],
            to=cfg["trusted_contact"]
        )
        _last_alert_time = now
        print("[Alert] SMS sent via Twilio.")
        return True
    except Exception as e:
        print("[Alert] Twilio error:", e)
        return False

def send_alert(message: str, location: str = None):
    location_text = f" Location: {location}" if location else ""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[HerSafe ALERT] {timestamp} - {message}{location_text}"
    print("="*60)
    print(full_message)
    print("="*60)
    send_alert_via_twilio(full_message)

# -------------------------
# Detection heuristics
# -------------------------
def normalize_text(text: str) -> str:
    return text.lower()

def keyword_score(text: str, keywords: List[str]) -> int:
    text = normalize_text(text)
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score

def urgency_heuristic(transcript: str, rms: float = None) -> bool:
    ks = keyword_score(transcript, CONFIG["keywords"])
    if ks >= CONFIG["keyword_match_threshold"]:
        if keyword_score(transcript, CONFIG["urgent_indicators"]) > 0:
            print("[Heuristic] Urgent indicator word found along with keywords.")
            return True
        if rms and rms > CONFIG["rms_threshold"]:
            print(f"[Heuristic] Loud speech detected (rms={rms:.1f}) with keywords.")
            return True
        print("[Heuristic] Keywords found; flagging alert.")
        return True
    return False

# -------------------------
# Audio Processing
# -------------------------
def rms_from_audio_frame(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    count = len(frame_bytes) // 2
    if count == 0:
        return 0.0
    import struct
    fmt = "<" + ("h" * count)
    samples = struct.unpack(fmt, frame_bytes[:count*2])
    ssum = sum([s*s for s in samples])
    mean_sq = ssum / count
    rms = math.sqrt(mean_sq)
    return rms

def listen_and_detect(audio_source="mic", audio_file=None, stop_event: threading.Event = None, out_queue: queue.Queue = None):
    r = sr.Recognizer()
    mic = None
    if audio_source == "mic":
        mic = sr.Microphone()
        print("[Voice] Using microphone. Please allow mic access if prompted.")
    elif audio_source == "file":
        if not audio_file or not os.path.exists(audio_file):
            raise FileNotFoundError("audio_file not found")
        audio = sr.AudioFile(audio_file)
    else:
        raise ValueError("audio_source must be 'mic' or 'file'")

    if mic:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1)
            print("[Voice] Calibrated for ambient noise.")

    while not (stop_event and stop_event.is_set()):
        try:
            if mic:
                with mic as source:
                    print("[Voice] Listening for up to 6 seconds...")
                    audio_data = r.listen(source, timeout=6, phrase_time_limit=6)
                rms_val = rms_from_audio_frame(audio_data.get_raw_data())
                try:
                    transcript = r.recognize_google(audio_data)
                except sr.UnknownValueError:
                    transcript = ""
                except sr.RequestError as e:
                    transcript = ""
                    print("[Voice] Speech API error:", e)
            else:
                with audio as source:
                    audio_data = r.record(source)
                rms_val = rms_from_audio_frame(audio_data.get_raw_data())
                try:
                    transcript = r.recognize_google(audio_data)
                except Exception:
                    transcript = ""
                if stop_event:
                    stop_event.set()

            print(f"[Voice] Transcript: '{transcript}' (rms={rms_val:.1f})")
            if transcript and urgency_heuristic(transcript, rms=rms_val):
                location = get_approx_location()
                send_alert(f"Detected distress speech: '{transcript}'", location)
                if out_queue:
                    out_queue.put(("voice", transcript, location))
            time.sleep(0.5)
        except sr.WaitTimeoutError:
            continue
        except KeyboardInterrupt:
            print("[Voice] KeyboardInterrupt - stopping listener.")
            if stop_event:
                stop_event.set()
            break
        except Exception as e:
            print("[Voice] Unexpected error:", e)
            time.sleep(1)

# -------------------------
# Chat/Text Monitoring
# -------------------------
def monitor_chat_messages(message_source="file", file_path=None, stop_event: threading.Event = None, out_queue: queue.Queue = None):
    print("[Chat] Chat monitor started.")
    seen_lines = 0
    simulated_messages = [
        "Hey, are you free today?",
        "I'm walking home now, hope it's safe",
        "I think someone is following me",
        "please help me! he's behind me",
    ]

    while not (stop_event and stop_event.is_set()):
        try:
            messages = []
            if message_source == "file":
                if not file_path or not os.path.exists(file_path):
                    time.sleep(CONFIG["chat_poll_interval"])
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                messages = lines[seen_lines:]
                seen_lines = len(lines)
            else:
                if simulated_messages:
                    messages = [simulated_messages.pop(0)]
                else:
                    messages = []

            for msg in messages:
                print(f"[Chat] New message: '{msg}'")
                if urgency_heuristic(msg):
                    location = get_approx_location()
                    send_alert(f"Detected distress text: '{msg}'", location)
                    if out_queue:
                        out_queue.put(("chat", msg, location))
            time.sleep(CONFIG["chat_poll_interval"])
        except KeyboardInterrupt:
            if stop_event:
                stop_event.set()
            break
        except Exception as e:
            print("[Chat] Error:", e)
            time.sleep(1)

# -------------------------
# Placeholder Location
# -------------------------
def get_approx_location():
    return "ApproxLat:12.9716, ApproxLong:77.5946"

# -------------------------
# Main Demo
# -------------------------
def main_demo(run_duration_seconds: int = 120, use_mic: bool = True):
    print("[Main] Initializing HerSafe prototype demo.")
    if init_twilio():
        print("[Main] Twilio configured: SMS enabled.")
    else:
        print("[Main] Twilio not enabled. Alerts will print to console.")

    stop_event = threading.Event()
    q = queue.Queue()

    voice_thread = threading.Thread(
        target=listen_and_detect,
        kwargs={"audio_source": "mic" if use_mic else "file", "audio_file": None, "stop_event": stop_event, "out_queue": q},
        daemon=True
    )
    chat_thread = threading.Thread(
        target=monitor_chat_messages,
        kwargs={"message_source": "list", "stop_event": stop_event, "out_queue": q},
        daemon=True
    )

    voice_thread.start()
    chat_thread.start()

    start = time.time()
    try:
        while time.time() - start < run_duration_seconds and not stop_event.is_set():
            try:
                src, txt, loc = q.get(timeout=1)
                print(f"[Main] Event from {src}: '{txt}' at {loc}")
            except queue.Empty:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received - stopping demo.")
    finally:
        stop_event.set()
        voice_thread.join(timeout=2)
        chat_thread.join(timeout=2)
        print("[Main] Demo finished.")

if __name__ == "__main__":
    main_demo(run_duration_seconds=120, use_mic=True)
