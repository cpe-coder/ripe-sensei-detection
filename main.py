import cv2
import numpy as np
import urllib.request
import time
import firebase_admin
from firebase_admin import credentials, db
from collections import deque

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aquaflow-b5a38-default-rtdb.firebaseio.com/'
})
detection_ref = db.reference('detection')
detection_ref.set({"ripe": 0, "raw": 0})

url = 'http://192.168.43.249/cam-hi.jpg'

ripe_count = 0
raw_count = 0
recent_centers = deque(maxlen=50)

def classify_tomato_color(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    if (35 < h < 85 and s > 50) or (v > 180 and s < 40):  # Green or white = raw
        return "raw"
    elif (h < 15 or h > 160) and s > 100:  # Red
        return "ripe"
    elif (15 <= h <= 35) and s > 100:  # Yellow
        return "ripe"
    return None

def find_tomatoes(frame):
    global ripe_count, raw_count

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_yellow = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    combined_mask = mask_red | mask_yellow | mask_green

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        if any(np.linalg.norm(np.array([cx, cy]) - np.array(rc)) < 30 for rc in recent_centers):
            continue

        recent_centers.append((cx, cy))

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        avg_color = cv2.mean(roi)[:3]
        label = classify_tomato_color(np.uint8(avg_color))

        if label == "ripe":
            ripe_count += 1
            color = (0, 0, 255)
        elif label == "raw":
            raw_count += 1
            color = (0, 255, 0)
        else:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    detection_ref.set({
        "ripe": ripe_count,
        "raw": raw_count
    })

try:
    while True:
        try:
            img_resp = urllib.request.urlopen(url, timeout=3)
            img_bytes = img_resp.read()
            imgnp = np.array(bytearray(img_bytes), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            if frame is None:
                print("⚠️ Failed to decode image.")
                continue

            find_tomatoes(frame)

            cv2.putText(frame, f"Ripe: {ripe_count}  Raw: {raw_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Tomato Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"⚠️ Error fetching image: {e}")
            time.sleep(1)

except KeyboardInterrupt:
    print("\n🔴 Stopped by user.")

finally:
    print("🛑 Cleaning up...")
    cv2.destroyAllWindows()
