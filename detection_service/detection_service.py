import pika
import cv2
import numpy as np
import os
import requests
import yaml
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

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
hand_tracker = DeepSort(max_age=30)
scooper_tracker = DeepSort(max_age=30)

# --- STATE ---
hand_states = {}  # hand_id: state info
violation_count = 0

# --- UTILS ---
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def in_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    cx, cy = bbox_center(bbox)
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

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

def update_streaming_service(violations):
    try:
        requests.post(STREAMING_SERVICE_URL, json={"violations": violations})
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
CONF_THRESH_HAND = 0.1  # Lowered threshold for hand detection
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
    global violation_count
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
        cv2.putText(frame, 'ROI', (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    send_frame_to_stream(frame)

    # --- Track hands and scoopers ---
    hand_detections = to_detections(hands)
    scooper_detections = to_detections(scoopers)
    hand_tracks = hand_tracker.update_tracks(hand_detections, frame=frame)
    scooper_tracks = scooper_tracker.update_tracks(scooper_detections, frame=frame)
    scooper_bboxes = [tuple(t.to_ltrb()) for t in scooper_tracks if t.is_confirmed()]

    # --- Stateful, sequential violation logic ---
    for track in hand_tracks:
        if not track.is_confirmed():
            continue
        hand_id = track.track_id
        hand_bbox = tuple(track.to_ltrb())
        # Get or initialize state for this hand
        state = hand_states.get(hand_id, {"was_in_roi": False})
        # Check intersection with any ROI
        in_roi = any(iou(hand_bbox, (roi['x1'], roi['y1'], roi['x2'], roi['y2'])) > 0.1 for roi in ROIS)
        if in_roi:
            state["was_in_roi"] = True
        # Check intersection with any pizza
        on_pizza = any(iou(hand_bbox, pizza) > 0.1 for pizza in pizzas)
        # Check if hand is holding a scooper
        holding_scooper = any(iou(hand_bbox, s) > 0.1 for s in scooper_bboxes)
        # Violation: hand was in ROI, now on pizza, and not holding scooper
        if on_pizza and state["was_in_roi"]:
            if not holding_scooper:
                print(f"[Violation] Hand {hand_id} touched pizza after ROI without scooper!")
                violation_count += 1
                update_streaming_service(violation_count)
            else:
                print(f"[Hand {hand_id}] used scooper correctly after ROI.")
            # Reset after pizza touch
            state["was_in_roi"] = False
        hand_states[hand_id] = state

    print(f"[Frame {frame_id}] Violations: {violation_count}")

# --- ENTRYPOINT ---
if __name__ == "__main__":
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
    connection, channel = connect_rabbitmq()
    channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback, auto_ack=True)
    print("Detection service running... Press Ctrl+C to stop.")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Stopping detection service...")
        connection.close()

