import cv2
import os
import pandas as pd
import time
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# ==========================
# CONFIG (edit paths if needed)
# ==========================
MODEL_PATH = "best.pt"          # YOLOv8 model
VIDEO_PATH = "data/test_video.mp4"      # input video
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"
CONF_THRESHOLD = 0.4
MAX_AGE = 60                    # allow longer so tracks don't die quickly
IOU_RECONNECT_THRESH = 0.5      # IoU threshold to reconnect to previous canonical id
CLASS_IOU_MATCH_MIN = 0.25      # min IoU for class assignment
FPS_FALLBACK = 25.0

# ==========================
# SETUP
# ==========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=MAX_AGE)  # we rely on deep sort + our reconnect logic

base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"{base_name}_tracked_{timestamp}.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

# ==========================
# CLASS MAPPING (your confirmed classes)
# ==========================
CLASS_MAP = {
    0: "Not Available",
    1: "Not Working",
    2: "Roaming",
    3: "Sleeping",
    4: "Talking",
    5: "Working"
}

# ==========================
# STATE STRUCTURES
# employee_data keyed by canonical_id (int):
# {
#   "last_status": str,
#   "work_sec": float,
#   "idle_sec": float,
#   "last_box": [x1,y1,x2,y2],
#   "last_cls": int,
#   "last_seen_frame": int
# }
# track_to_canonical maps deep-sort track_id -> canonical_id
# ==========================
employee_data = {}
track_to_canonical = {}
next_canonical_id = 1
frame_idx = 0
frame_dt = 1.0 / fps

print("[INFO] Starting tracking...")

# ==========================
# helper functions
# ==========================
def iou_xyxy(boxA, boxB):
    # boxes = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union

def format_hhmmss(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # run YOLO on the frame
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False , save=False, save_txt=False, project=None, name=None )
    detections = []  # list of dicts: {"xyxy":[x1,y1,x2,y2], "conf":float, "cls":int}

    if len(results):
        boxes = results[0].boxes
        if boxes is not None and boxes.xyxy is not None:
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for xyxy, conf, cls in zip(xyxys, confs, clss):
                x1, y1, x2, y2 = map(float, xyxy)
                detections.append({
                    "xyxy": [x1, y1, x2, y2],
                    "conf": float(conf),
                    "cls": int(cls)
                })

    # prepare for deep sort: bbox as [x,y,w,h]
    det_list = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        w, h = x2 - x1, y2 - y1
        det_list.append(([x1, y1, w, h], d["conf"], int(d["cls"])))

    # update deep sort (returns Track objects)
    tracks = tracker.update_tracks(det_list, frame=frame)

    # we'll collect which canonical IDs were seen this frame
    seen_canonical_ids = set()

    # Process each track
    for t in tracks:
        if not t.is_confirmed():
            continue

        ds_track_id = t.track_id
        ltrb = t.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)
        track_box = [x1, y1, x2, y2]

        # find best detection class for this track by IoU
        best_cls = None
        best_iou = 0.0
        for d in detections:
            i = iou_xyxy(track_box, d["xyxy"])
            if i > best_iou:
                best_iou = i
                best_cls = d["cls"]

        detected_status = CLASS_MAP.get(best_cls, "Unknown") if best_iou >= CLASS_IOU_MATCH_MIN else "Unknown"

        # Determine canonical id for this deep-sort track:
        canonical_id = track_to_canonical.get(ds_track_id)

        # If this track isn't mapped yet, try to reconnect to existing canonical using IoU & class
        if canonical_id is None:
            matched_cid = None
            for cid, emp in employee_data.items():
                # if last_box exists, compute IoU between current track_box and stored last_box
                if "last_box" in emp and emp["last_box"] is not None:
                    i = iou_xyxy(track_box, emp["last_box"])
                    # require class match (best_cls) if available and IoU threshold
                    if i >= IOU_RECONNECT_THRESH:
                        # if we have a previous class recorded (last_cls), prefer same class
                        if emp.get("last_cls") is None or best_cls is None:
                            matched_cid = cid
                            break
                        elif emp.get("last_cls") == best_cls:
                            matched_cid = cid
                            break
            if matched_cid is not None:
                canonical_id = matched_cid
                track_to_canonical[ds_track_id] = canonical_id

        # If still no canonical id, create a new canonical id and map it
        if canonical_id is None:
            canonical_id = next_canonical_id
            next_canonical_id += 1
            track_to_canonical[ds_track_id] = canonical_id
            # initialize employee_data for this canonical id
            employee_data[canonical_id] = {
                "last_status": detected_status,
                "work_sec": 0.0,
                "idle_sec": 0.0,
                "last_box": track_box,
                "last_cls": best_cls,
                "last_seen_frame": frame_idx
            }
        else:
            # if canonical exists but previously stored under another track id, keep it and update fields
            if canonical_id not in employee_data:
                employee_data[canonical_id] = {
                    "last_status": detected_status,
                    "work_sec": 0.0,
                    "idle_sec": 0.0,
                    "last_box": track_box,
                    "last_cls": best_cls,
                    "last_seen_frame": frame_idx
                }

        emp = employee_data[canonical_id]

        # ===== TIME UPDATE LOGIC =====
        # We will add frame_dt to the PREVIOUS status holder for this canonical id.
        prev_status = emp.get("last_status", "Unknown")
        if prev_status.lower() == "working":
            emp["work_sec"] += frame_dt
        else:
            emp["idle_sec"] += frame_dt

        # Now update canonical entry to current detection for next frame
        emp["last_status"] = detected_status
        emp["last_box"] = track_box
        emp["last_cls"] = best_cls
        emp["last_seen_frame"] = frame_idx

        seen_canonical_ids.add(canonical_id)

        # ===== DRAW ON FRAME =====
        # Color: green if current detected_status == "Working" else red
        if detected_status.lower() == "working":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{canonical_id} {detected_status}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # live formatted counters
        work_str = format_hhmmss(emp["work_sec"])
        idle_str = format_hhmmss(emp["idle_sec"])
        cv2.putText(frame, f"W:{work_str}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.putText(frame, f"N:{idle_str}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

    # optionally: age-out mapping entries for tracks that disappeared: we keep canonical data (so times preserved)
    # write and show
    out_vid.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# FINALIZE: Save CSV summary
# ==========================
today_str = datetime.now().strftime("%Y-%m-%d")
csv_path = os.path.join(LOG_DIR, f"activity_log_{today_str}.csv")

rows = []
for cid, emp in employee_data.items():
    rows.append({
        "ID": cid,
        "Last Status": emp.get("last_status", "Unknown"),
        "Working Time": format_hhmmss(emp.get("work_sec", 0.0)),
        "Not Working Time": format_hhmmss(emp.get("idle_sec", 0.0)),
        "Last Seen Frame": emp.get("last_seen_frame", 0)
    })

pd.DataFrame(rows).to_csv(csv_path, index=False)

cap.release()
out_vid.release()
cv2.destroyAllWindows()

print("[DONE]")
print("Video saved:", output_path)
print("CSV saved:", csv_path)
