import pika
import cv2
import numpy as np
import os
import requests
import yaml
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from violation_db import save_violation

VIOLATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'shared', 'violations')
os.makedirs(VIOLATION_DIR, exist_ok=True)

# global VideoWriter and output path
alert_video_writer = None
alert_video_path = None
#  global for alert banner duration
alert_frames_remaining = 0  # Number of frames to keep alert visible
ALERT_DURATION_FRAMES = 10  # Show alert for 1 second at 25 FPS
VIOLATION_COOLDOWN_FRAMES = 50  # Number of frames to wait before allowing another violation for the same hand
SCOOPER_GRACE_FRAMES = 50  # Number of frames after releasing scooper during which no violation is triggered

# --- CONFIGURATION ---
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'localhost')
RABBITMQ_QUEUE = os.environ.get('RABBITMQ_QUEUE', 'video_frames')
MODEL_PATH = os.environ.get('MODEL_PATH', 'shared/model/yolo12m-v2.pt')
STREAMING_SERVICE_URL = os.environ.get('STREAMING_SERVICE_URL', 'http://localhost:8000/metadata')
FRAME_STREAM_URL = os.environ.get('FRAME_STREAM_URL', 'http://localhost:8000/frame')
ROI_CONFIG_PATH = os.environ.get('ROI_CONFIG_PATH', 'shared/roi_config.yaml')

# --- LOAD ROIS ---
def load_rois():
    import yaml
    with open(ROI_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('rois', [])

ROIS = load_rois()

# --- TRACKERS ---
# Tune DeepSORT: hand_tracker with higher max_age and n_init
hand_tracker = DeepSort(max_age=60, n_init=2)
scooper_tracker = DeepSort(max_age=30)

# --- STATE ---
hand_states = {}  # hand_id: state info
violation_count = 0
# Buffer for recently lost hands to handle ID switches
recently_lost_hands = {}  # hand_id: {"state": ..., "last_bbox": ..., "last_seen_frame": ...}
RECENTLY_LOST_MAX_AGE = 20  # frames 
SCOOPER_GRACE_FRAMES = 10  # Number of frames after releasing scooper during which no violation is triggered

# --- UTILS ---
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def is_center_in_roi(hand_bbox, roi):
    cx, cy = bbox_center(hand_bbox)
    return roi['x1'] <= cx <= roi['x2'] and roi['y1'] <= cy <= roi['y2']

def in_any_roi(bbox, rois):
    x1, y1, x2, y2 = bbox
    cx, cy = bbox_center(bbox)
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
    return False

def is_close(bbox1, bbox2, thresh=50):
    c1 = bbox_center(bbox1)
    c2 = bbox_center(bbox2)
    return np.linalg.norm(np.array(c1) - np.array(c2)) < thresh

def hand_near_pizza(hand_bbox, pizzas):
    for pizza in pizzas:
        if is_close(hand_bbox, pizza, thresh=80):
            return True
    return False

def update_streaming_service(violations, bboxes=None, labels=None, violation_status=None):
    try:
        payload = {"violations": violations}
        if bboxes is not None:
            payload["bboxes"] = bboxes
        if labels is not None:
            payload["labels"] = labels
        if violation_status is not None:
            payload["violation_status"] = violation_status
        requests.post(STREAMING_SERVICE_URL, json=payload)
    except Exception as e:
        print("Could not update streaming service:", e)

def send_frame_to_stream(frame):
    try:
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        requests.post(FRAME_STREAM_URL, data=frame_bytes, headers={"Content-Type": "image/jpeg"})
    except Exception    as e:
        print("Could not send frame to stream:", e)

# --- SAFELY FORMAT DETECTIONS ---
def to_detections(arr, label="object"):
    result = []
    if arr is None:
        return result
    arr = np.array(arr)
    if arr.size == 0:
        return result
    if isinstance(arr, (float, int, np.floating, np.integer)):
        return result
    if arr.ndim == 1 and arr.shape[0] == 4:
        x1, y1, x2, y2 = arr
        w, h = x2 - x1, y2 - y1
        result.append(([float(x1), float(y1), float(w), float(h)], 1.0, label))
    elif arr.ndim == 2:
        for box in arr:
            if hasattr(box, '__iter__') and len(box) == 4:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                result.append(([float(x1), float(y1), float(w), float(h)], 1.0, label))
    elif isinstance(arr, list):
        for box in arr:
            if hasattr(box, '__iter__') and len(box) == 4:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                result.append(([float(x1), float(y1), float(w), float(h)], 1.0, label))
    return result

# --- Class mapping for readability ---
CLASS_HAND = 0
CLASS_PERSON = 1
CLASS_PIZZA = 2
CLASS_SCOOPER = 3
CONF_THRESH_HAND = 0.05  # Lowered threshold for hand detection
CONF_THRESH_PIZZA = 0.2
CONF_THRESH_SCOOPER = 0.05  # Lower threshold for scooper

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH

def sufficient_overlap(boxA, boxB, min_ratio=0.1):
    inter_area = intersection_area(boxA, boxB)
    hand_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return hand_area > 0 and (inter_area / hand_area) >= min_ratio

def rects_overlap(boxA, boxB):
    """Return True if two rectangles (x1, y1, x2, y2) overlap at all."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

# --- RABBITMQ CONNECTION ---
def connect_rabbitmq():
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE)
    return connection, channel

def preprocess_frame(frame):
    # 1. Convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Gamma correction
    gamma = 1.2  # >1 brightens, <1 darkens
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])).astype("uint8")
    frame = cv2.LUT(frame, table)

    # 3. Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)

    return frame

# --- CALLBACK FUNCTION ---
def callback(ch, method, properties, body):
    global violation_count, alert_video_writer, alert_video_path, alert_frames_remaining, recently_lost_hands
    frame_id = properties.headers.get('frame_id', -1)
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Use the original frame for detection
    results = model(frame)[0]

    # --- Filter detections by confidence ---
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    hands, scoopers, pizzas = [], [], []
    for box, conf, cls in zip(boxes, confs, clss):
        label = int(cls)
        if label == CLASS_HAND and conf >= CONF_THRESH_HAND:
            hands.append(box)
        elif label == CLASS_PIZZA and conf >= CONF_THRESH_PIZZA:
            pizzas.append(box)
        elif label == CLASS_SCOOPER and conf >= CONF_THRESH_SCOOPER:
            scoopers.append(box)

    # --- Draw detections ---
    for box in hands:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, 'hand', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    for box in scoopers:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame, 'scooper', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    for box in pizzas:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(frame, 'pizza', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    # Draw all ROIs
    for roi in ROIS:
        rx1, ry1, rx2, ry2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0,255,255), 2)
        roi_name = roi.get('name', 'ROI')
        cv2.putText(frame, roi_name, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    send_frame_to_stream(frame)

    # --- Track hands and scoopers ---
    hand_detections = to_detections(hands)
    scooper_detections = to_detections(scoopers)
    hand_tracks = hand_tracker.update_tracks(hand_detections, frame=frame)
    scooper_tracks = scooper_tracker.update_tracks(scooper_detections, frame=frame)
    scooper_bboxes = [tuple(t.to_ltrb()) for t in scooper_tracks if t.is_confirmed()]

    # --- Identify which hands are holding a scooper ---
    hands_with_scooper = set()
    for track in hand_tracks:
        if not track.is_confirmed():
            continue
        hand_id = track.track_id
        hand_bbox = tuple(track.to_ltrb())
        if any(iou(hand_bbox, s) > 0.1 for s in scooper_bboxes):
            hands_with_scooper.add(hand_id)

    # --- Stateful, sequential violation logic ---
    MAX_MISSING_FRAMES = 5  # Temporal smoothing for ghost tracks
    active_hand_ids = set()
    violation_in_this_frame = False
    # --- Check if any hand is holding a scooper this frame ---
    any_hand_holding_scooper = False
    for track in hand_tracks:
        if not track.is_confirmed():
            continue
        hand_bbox = tuple(track.to_ltrb())
        if any(iou(hand_bbox, s) > 0.1 for s in scooper_bboxes):
            any_hand_holding_scooper = True
            break
    # --- Try to match new tracks to recently lost hands ---
    for track in hand_tracks:
        if not track.is_confirmed():
            continue  # Only match for confirmed tracks
        hand_id = track.track_id
        hand_bbox = tuple(track.to_ltrb())
        # Try to use appearance features if available
        new_feat = getattr(track, 'feature', None)
        if hand_id not in hand_states:
            best_match_id = None
            best_dist = float('inf')
            best_feat_dist = float('inf')
            best_reason = None
            for lost_id, lost_info in recently_lost_hands.items():
                lost_bbox = lost_info["last_bbox"]
                dist = np.linalg.norm(np.array(bbox_center(hand_bbox)) - np.array(bbox_center(lost_bbox)))
                age = frame_id - lost_info["last_seen_frame"] if frame_id >= 0 else 0
                # Appearance feature distance if available
                lost_feat = lost_info.get("feature", None)
                feat_dist = None
                if new_feat is not None and lost_feat is not None:
                    # Use cosine distance for appearance features
                    feat_dist = 1 - np.dot(new_feat, lost_feat) / (np.linalg.norm(new_feat) * np.linalg.norm(lost_feat) + 1e-6)
                # Prefer feature match if available and close
                if feat_dist is not None and feat_dist < 0.4 and age <= RECENTLY_LOST_MAX_AGE:
                    if feat_dist < best_feat_dist:
                        best_feat_dist = feat_dist
                        best_match_id = lost_id
                        best_reason = f"appearance (feat_dist={feat_dist:.3f}, age={age})"
                # Otherwise, fallback to spatial match (increase threshold to 150)
                elif dist < 150 and age <= RECENTLY_LOST_MAX_AGE:
                    if dist < best_dist:
                        best_dist = dist
                        best_match_id = lost_id
                        best_reason = f"spatial (dist={dist:.1f}, age={age})"
            if best_match_id is not None:
                print(f"[DEBUG] Transferring state from lost hand {best_match_id} to new hand {hand_id} ({best_reason})")
                # Always transfer full state, including grace period
                hand_states[hand_id] = recently_lost_hands[best_match_id]["state"].copy()
                del recently_lost_hands[best_match_id]
            else:
                print(f"[DEBUG] No match found for new hand {hand_id} (bbox={hand_bbox})")
    for track in hand_tracks:
        if not track.is_confirmed():
            # Temporal smoothing: keep state for ghost tracks
            hand_id = track.track_id
            state = hand_states.get(hand_id)
            if state is not None:
                state['missing_frames'] = state.get('missing_frames', 0) + 1
                if state['missing_frames'] <= MAX_MISSING_FRAMES:
                    # Treat as still present
                    hand_states[hand_id] = state
                    continue
                else:
                    # Before removing, store in recently_lost_hands
                    last_bbox = getattr(track, 'last_bbox', None)
                    if last_bbox is None:
                        # Try to get from state if stored
                        last_bbox = state.get('last_bbox', None)
                    # Try to store appearance feature if available
                    last_feat = getattr(track, 'feature', None)
                    if last_bbox is None:
                        # Fallback: skip storing
                        hand_states.pop(hand_id, None)
                        continue
                    recently_lost_hands[hand_id] = {
                        "state": state,
                        "last_bbox": last_bbox,
                        "last_seen_frame": frame_id if frame_id >= 0 else 0,
                        "feature": last_feat
                    }
                    print(f"[DEBUG] Stored lost hand {hand_id} (bbox={last_bbox}, feature={'yes' if last_feat is not None else 'no'})")
                    hand_states.pop(hand_id, None)
            continue
        hand_id = track.track_id
        hand_bbox = tuple(track.to_ltrb())
        # Save last_bbox in state for lost hand recovery
        state = hand_states.get(hand_id, {"was_in_roi": False, "missing_frames": 0, "must_leave_roi": False, "violation_cooldown": 0, "must_leave_pizza_and_roi": False, "recently_had_scooper": 0})
        state['last_bbox'] = hand_bbox
        # Check intersection with any ROI (using center only)
        in_roi = any(is_center_in_roi(hand_bbox, roi) for roi in ROIS)
        # Decrement cooldown if active
        if state.get('violation_cooldown', 0) > 0:
            state['violation_cooldown'] -= 1
        # Check intersection with any pizza (using minimum overlap, now 5%)
        on_pizza = any(sufficient_overlap(hand_bbox, pizza, min_ratio=0.05) for pizza in pizzas)
        # Check if this hand is holding a scooper
        holding_scooper = hand_id in hands_with_scooper
        if holding_scooper:
            state["recently_had_scooper"] = SCOOPER_GRACE_FRAMES  # reset grace counter
        elif state["recently_had_scooper"] > 0:
            state["recently_had_scooper"] -= 1  # countdown
        # Robust state: must leave both pizza and ROI after a violation
        if state.get('must_leave_pizza_and_roi', False):
            if not on_pizza and not in_roi:
                # Hand has left both pizza and ROI, re-arm
                state['must_leave_pizza_and_roi'] = False
                state['was_in_roi'] = False
            else:
                hand_states[hand_id] = state
                active_hand_ids.add(hand_id)
                continue
        # Violation: hand was in ROI, now on pizza, and not holding scooper
        if on_pizza and state["was_in_roi"]:
            if state.get('violation_cooldown', 0) == 0:
                # CHANGED: Skip violation if any hand holding scooper OR this hand recently had one
                if not any_hand_holding_scooper and state["recently_had_scooper"] == 0:
                    print(f"[Violation] Hand {hand_id} touched pizza after ROI without scooper!")
                    violation_count += 1
                    violation_in_this_frame = True
                    alert_frames_remaining = ALERT_DURATION_FRAMES  # Start/refresh alert duration
                    # Save violation frame
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    frame_filename = f"violation_{frame_id}_{hand_id}_{timestamp}.jpg"
                    frame_path = os.path.join(VIOLATION_DIR, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    # Save violation metadata
                    save_violation(
                        frame_path=frame_path,
                        bboxes=[hand_bbox],
                        labels=["hand"],
                        timestamp=timestamp
                    )
                    # Send results to streaming service
                    update_streaming_service(
                        violation_count,
                        bboxes=[hand_bbox],
                        labels=["hand"],
                        violation_status=True
                    )
                    # Set cooldown
                    state['violation_cooldown'] = VIOLATION_COOLDOWN_FRAMES
                    # Set must_leave_pizza_and_roi so next violation is not counted until hand leaves both
                    state['must_leave_pizza_and_roi'] = True
                else:
                    print(f"[Hand {hand_id}] used scooper correctly after ROI or grace period.")
                    update_streaming_service(
                        violation_count,
                        bboxes=[hand_bbox],
                        labels=["hand"],
                        violation_status=False
                    )
                    state["must_leave_roi"] = True
            state["was_in_roi"] = False
        # If hand is not on pizza, allow new violation after cooldown
        if not on_pizza:
            state['violation_cooldown'] = 0
        if in_roi:
            if state.get("must_leave_roi", False):
                # Must leave ROI before re-arming
                pass  # Do not set was_in_roi
            else:
                state["was_in_roi"] = True
        else:
            # If not in ROI, clear must_leave_roi
            state["must_leave_roi"] = False
        hand_states[hand_id] = state
        active_hand_ids.add(hand_id)

    # Clean up hand_states for tracks missing too long
    to_remove = []
    for hand_id, state in hand_states.items():
        if hand_id not in active_hand_ids:
            state['missing_frames'] = state.get('missing_frames', 0) + 1
            if state['missing_frames'] > MAX_MISSING_FRAMES:
                # Before removing, store in recently_lost_hands
                last_bbox = state.get('last_bbox', None)
                if last_bbox is not None:
                    recently_lost_hands[hand_id] = {
                        "state": state,
                        "last_bbox": last_bbox,
                        "last_seen_frame": frame_id if frame_id >= 0 else 0
                    }
                to_remove.append(hand_id)
    for hand_id in to_remove:
        hand_states.pop(hand_id, None)
    # Clean up old entries in recently_lost_hands
    if frame_id >= 0:
        to_remove_lost = [hid for hid, info in recently_lost_hands.items() if frame_id - info["last_seen_frame"] > RECENTLY_LOST_MAX_AGE]
        for hid in to_remove_lost:
            del recently_lost_hands[hid]

    # --- Save alert video (always enabled) ---
    if alert_video_writer is None:
        # Initialize VideoWriter on first frame
        h, w = frame.shape[:2]
        fps = 25  # Default FPS, adjust if you have actual FPS info
        alert_video_path = os.path.join(VIOLATION_DIR, f"alert_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        alert_video_writer = cv2.VideoWriter(alert_video_path, fourcc, fps, (w, h))
    frame_to_write = frame.copy()
    # Overlay violation count (top left corner, in red)
    count_text = f"Violations: {violation_count}"
    cv2.putText(frame_to_write, count_text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    # Overlay alert banner if violation or alert is active
    if alert_frames_remaining > 0:
        cv2.rectangle(frame_to_write, (0, 0), (frame_to_write.shape[1], 80), (0, 0, 255), -1)  # Red banner
        cv2.putText(frame_to_write, "VIOLATION DETECTED!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        alert_frames_remaining -= 1
    alert_video_writer.write(frame_to_write)

    print(f"[Frame {frame_id}] Violations: {violation_count}")

# --- ENTRYPOINT ---
if __name__ == "__main__":
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
    connection, channel = connect_rabbitmq()
    # Purge the queue before starting to consume
    try:
        channel.queue_purge(queue=RABBITMQ_QUEUE)
        print(f"Purged queue: {RABBITMQ_QUEUE}")
    except Exception as e:
        print(f"Warning: Could not purge queue {RABBITMQ_QUEUE}: {e}")
    channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback, auto_ack=True)
    print("Detection service running... Press Ctrl+C to stop.")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Stopping detection service...")
        if alert_video_writer is not None:
            alert_video_writer.release()
            print(f"Alert video saved to: {alert_video_path}")
        connection.close()