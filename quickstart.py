import requests
import numpy as np

capture_url = f"http://172.29.104.71/"

session = requests.Session()  # 🔥 faster repeated requests

while True:
    try:
        response = session.get(capture_url, timeout=2)

        if response.status_code != 200:
            continue

        img_array = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # ===== ML =====
        results = model.track(frame, imgsz=320, conf=conf, persist=True, verbose=False)[0]

        if results.boxes is not None and results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            classes   = results.boxes.cls.int().cpu().tolist()
            boxes     = results.boxes.xyxy.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box

                if track_id not in object_memory:
                    if cls_id == 0:
                        object_memory[track_id] = {"type": "HUMAN", "color": (0,255,0)}
                    elif cls_id == 58:
                        object_memory[track_id] = {"type": "BANK", "color": (30,180,30)}
                    else:
                        object_memory[track_id] = {"type": "TRASH", "color": (0,0,255)}

                data = object_memory[track_id]

                if arduino and data["type"] == "TRASH":
                    arduino.write(b'G')

                cv2.rectangle(frame, (x1,y1), (x2,y2), data["color"], 2)
                cv2.putText(frame, f"{data['type']} {track_id}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            data["color"], 2)

        cv2.imshow("AQUA Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except requests.exceptions.RequestException:
        print("⚠ Frame request failed, retrying...")
        continue
