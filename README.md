# Pizza Store Scooper Violation Detection System

---

## Project Overview
This project is a microservices-based computer vision system for monitoring hygiene protocol compliance in a pizza store. It detects whether workers use a scooper when picking up ingredients from critical zones (ROIs) and flags violations in real time.

---

## Features
- Reads video frames from file or camera
- Detects hands, scoopers, pizzas, and persons using YOLO
- Tracks objects with DeepSORT
- Supports multiple user-defined ROIs (e.g., protein containers)
- Flags violations only if a hand enters an ROI and then later touches the pizza without holding a scooper. If the hand is holding a scooper when touching the pizza, no violation is flagged. After a correct scooper use (touches pizza with scooper), the hand must leave the ROI before a new violation can be triggered. While in the ROI after correct use, no violation is possible until the hand leaves and re-enters the ROI. Each hand is tracked independently across frames.
- Lowered hand detection threshold for higher recall, improving detection of small or partially occluded hands.
- Tuned DeepSORT tracking parameters (higher max_age, min_hits) for more robust hand tracking, especially with multiple workers.
- Temporal smoothing: hand tracks are kept alive for a few frames even if missed, handling short occlusions and improving multi-worker tracking.
- Streams annotated video and violation count to a web frontend
- Real-time alarm: The frontend displays a red warning box and violation details (bounding boxes, labels) when a violation is detected
- Modular microservices architecture (frame reader, detection, streaming, frontend)

---

## Microservices Overview

### 1. Frame Reader Service (`frame_reader/`)
**Role:** Reads video frames from a file or RTSP camera feed and publishes them to a message broker (RabbitMQ).
- **Technologies:** Python, OpenCV, pika
- **How it works:** Reads frames, publishes each as a JPEG to RabbitMQ, decoupling video ingestion from detection.

### 2. Detection Service (`detection_service/`)
**Role:** Subscribes to the message broker, performs object detection and tracking, applies violation logic, and updates the streaming service.
- **Technologies:** Python, Ultralytics YOLO, DeepSORT, OpenCV, pika, requests
- **Logic Summary:**
    - Each hand is tracked independently across frames.
    - If a hand enters an ROI, it is marked as "was in ROI".
    - When that hand later touches the pizza:
        - If holding a scooper: **No violation** (and the hand must leave the ROI before a new violation can be triggered)
        - If not holding a scooper: **Violation flagged**
    - After a correct scooper use (touches pizza with scooper), the hand must leave the ROI before a new violation can be triggered. While in the ROI after correct use, no violation is possible until the hand leaves and re-enters the ROI.
- **Robustness Improvements:**
    - Lowered hand detection threshold for higher recall.
    - DeepSORT tracker tuned with higher max_age and min_hits for better continuity.
    - Temporal smoothing: hand tracks are kept alive for a few frames even if missed, handling short occlusions and improving multi-worker tracking.
- **How it works:** Receives frames, runs YOLO, tracks hands/scoopers, checks ROIs, flags violations, sends results to streaming service.

### 3. Streaming Service (`streaming_service/`)
**Role:** Serves detection results and video frames to the frontend via REST API and WebSocket.
- **Technologies:** Python, FastAPI, Uvicorn, WebSocket
- **How it works:** Receives violation count and frames, provides REST/WebSocket endpoints for frontend.

### 4. Frontend UI (`frontend/`)
**Role:** Visualizes the video stream, bounding boxes, ROIs, and violation events for the user.
- **Technologies:** Streamlit, requests
- **How it works:** Connects to streaming service, displays frames and violation count in real time.
- **Alarm Feature:** When a violation is detected, a red warning box appears in the UI, along with bounding boxes and labels for the violation. The alarm clears automatically when no violation is present.

### 5. Shared Resources (`shared/`)
**Role:** Stores all shared data, models, configs, and videos used by the system.
- **Contents:**
  - `dataset/`: Training/validation data (if retraining YOLO)
  - `model/`: Pretrained YOLO weights
  - `videos/`: Test and input videos
  - `violations/`: Saving the frame in which violation happens.
  - `roi_config.yaml`: User-defined ROIs for detection

---

## Project Structure
```
pizza_violation_system/
├── frame_reader/
├── detection_service/
├── streaming_service/
├── frontend/
├── shared/
├── select_rois.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Quick Start
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd pizza_violation_system
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install and start RabbitMQ:**
   - Download from https://www.rabbitmq.com/download.html
   - Start the RabbitMQ service
5. **Prepare resources:**
   - Place your test videos in `shared/videos/`
   - Place your YOLO model weights in `shared/model/`
6. **Define ROIs:**
   - Run the interactive script to select up to 10 ROIs on the first frame of your video:
     ```bash
     python select_rois.py
     ```
   - (You can set the video path in the script or with the `VIDEO_PATH` environment variable.)
7. **Run the system (in four terminals):**
   - Streaming service: `python streaming_service/streaming_service.py`
   - Detection service: `python detection_service/detection_service.py`
   - Frame reader: `python frame_reader/frame_reader.py "shared/videos/YourVideo.mp4"`
   - Frontend: `streamlit run frontend/app.py`

---

### Choosing ROIs (Regions of Interest)
- The ROI selection script (`select_rois.py`) will display the first frame of your video.
- Draw up to 8 rectangles with your mouse, each covering a critical ingredient/protein bin.
- Press `q` when done. The coordinates are saved to `shared/roi_config.yaml`.
- Make sure the video path in the script matches your detection video.
- Restart the detection service after updating ROIs.

**Tip:** The order you draw the ROIs is the order they appear in the config file.

---

## Example ROI Config (`shared/roi_config.yaml`)
```yaml
rois:
  - {x1: 100, y1: 200, x2: 200, y2: 300}
  - {x1: 210, y1: 200, x2: 310, y2: 300}
  - {x1: 320, y1: 200, x2: 420, y2: 300}
  - {x1: 430, y1: 200, x2: 530, y2: 300}
  - {x1: 540, y1: 200, x2: 640, y2: 300}
  - {x1: 650, y1: 200, x2: 750, y2: 300}
  - {x1: 760, y1: 200, x2: 860, y2: 300}
  - {x1: 870, y1: 200, x2: 970, y2: 300}
```

---

## Violation Frame Storage
- When a violation is detected, the corresponding video frame is saved as an image in the `shared/violations/` directory. Each file is named with the violation details (frame ID, hand ID, timestamp).
- This directory is created automatically if it does not exist.

---

## .gitignore Note
- The `shared/` directory (datasets, models, videos, ROI configs) and `venv/` (virtual environment) are **ignored by git** and will not be committed, except for `shared/roi_config.yaml` and `shared/model/`.
- If you want to share a sample ROI config, copy it to a different location or provide an example in the README.

---

## Example: Extract a Frame for ROI Selection
```python
import cv2
cap = cv2.VideoCapture('shared/videos/YourVideo.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('SampleFrame.png', frame)
cap.release()
```

---

## Example: Set Video Path for ROI Selection
```bash
set VIDEO_PATH=shared/videos/YourVideo.mp4  # On Windows
# or
export VIDEO_PATH=shared/videos/YourVideo.mp4  # On Mac/Linux
python select_rois.py
```

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License
MIT License 
