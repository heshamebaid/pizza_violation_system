# Pizza Store Scooper Violation Detection System

This project is a microservices-based computer vision system for monitoring hygiene protocol compliance in a pizza store. It detects whether workers use a scooper when picking up ingredients from critical zones (ROIs) and flags violations in real time.

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
   - Run the interactive script to select up to 8 ROIs on the first frame of your video:
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

## .gitignore Note
- The `shared/` directory (datasets, models, videos, ROI configs) and `venv/` (virtual environment) are **ignored by git** and will not be committed.
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

### Choosing ROIs (Regions of Interest)

- The ROI selection script (`select_rois.py`) will display the first frame of your video.
- Draw up to 8 rectangles with your mouse, each covering a critical ingredient/protein bin.
- Press `q` when done. The coordinates are saved to `shared/roi_config.yaml`.
- Make sure the video path in the script matches your detection video.
- Restart the detection service after updating ROIs.

**Tip:** The order you draw the ROIs is the order they appear in the config file.

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License
MIT License 