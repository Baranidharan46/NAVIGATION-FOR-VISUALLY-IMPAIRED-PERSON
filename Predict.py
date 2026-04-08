import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading

# ─────────────────────────────────────────
# TEXT-TO-SPEECH — runs in background thread
# so it NEVER blocks the video frame
# ─────────────────────────────────────────
tts_lock = threading.Lock()

def speak_thread(command):
    with tts_lock:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            engine.setProperty('volume', 1.0)
            engine.say(command)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS ERROR] {e}")

def give_voice_command(command):
    print(f"[AUDIO] {command}")
    t = threading.Thread(target=speak_thread, args=(command,), daemon=True)
    t.start()


# ─────────────────────────────────────────
# MODEL SETUP
# yolov8m = best balance of speed + accuracy for real-time webcam
# ─────────────────────────────────────────
model = YOLO('yolov8m.pt')

DETECT_CLASS_IDS = {0, 1, 2, 3, 5, 7, 9, 11}

CLASS_SPEECH_NAMES = {
    0:  'person',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    5:  'bus',
    7:  'truck',
    9:  'traffic light',
    11: 'stop sign',
}

CONFIDENCE_THRESHOLD = 0.50


# ─────────────────────────────────────────
# ZONE LOGIC
# ─────────────────────────────────────────
def get_zone(x_center, frame_width):
    left_boundary  = frame_width // 3
    right_boundary = 2 * (frame_width // 3)
    if x_center < left_boundary:
        return 'left'
    elif x_center > right_boundary:
        return 'right'
    else:
        return 'center'

def get_direction_message(label, zone):
    if zone == 'left':
        return f"Caution! {label} detected on your left. Please move to the right."
    elif zone == 'right':
        return f"Caution! {label} detected on your right. Please move to the left."
    else:
        return f"Warning! {label} directly ahead. Please stop or slow down."


# ─────────────────────────────────────────
# COOLDOWN TRACKER
# ─────────────────────────────────────────
COOLDOWN_SECONDS = 4
last_spoken_time = {}

def can_speak(zone):
    now = time.time()
    if zone not in last_spoken_time:
        return True
    return (now - last_spoken_time[zone]) >= COOLDOWN_SECONDS

def mark_spoken(zone):
    last_spoken_time[zone] = time.time()


# ─────────────────────────────────────────
# WEBCAM SETUP
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Navigation system started. Press 'q' to quit.")

frame_count    = 0
SKIP_FRAMES    = 3
detected_zones = {}

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_count += 1

    if frame_count % SKIP_FRAMES == 0:
        detected_zones = {}
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls)
                confidence = float(box.conf)

                if class_id not in DETECT_CLASS_IDS:
                    continue
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                label = CLASS_SPEECH_NAMES[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2) // 2
                zone = get_zone(x_center, frame_width)

                if zone not in detected_zones or confidence > detected_zones[zone]['conf']:
                    detected_zones[zone] = {
                        'label': label,
                        'conf':  confidence,
                        'box':   (x1, y1, x2, y2),
                        'zone':  zone,
                    }

        for zone in ['center', 'left', 'right']:
            if zone in detected_zones and can_speak(zone):
                det = detected_zones[zone]
                message = get_direction_message(det['label'], det['zone'])
                give_voice_command(message)
                mark_spoken(zone)

    # Draw boxes on every frame (reuse last detections)
    for zone, det in detected_zones.items():
        x1, y1, x2, y2 = det['box']
        label      = det['label']
        confidence = det['conf']
        color = (0, 0, 255) if zone == 'center' else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.0%} [{zone}]",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )

    # Zone dividers
    cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 0), 1)
    cv2.putText(frame, 'LEFT',   (10, 30),                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, 'CENTER', (frame_width//2 - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, 'RIGHT',  (frame_width - 80, 30),    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('YOLOv8 Navigation System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Navigation system stopped.")