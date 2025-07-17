import cv2
import yaml

# Path to your video file
video_path = "shared/videos/Sah b3dha ghalt.mp4"  # Change to your video path

# Read the first frame from the video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame from video.")
    exit(1)

clone = frame.copy()
boxes = []
drawing = False
start_point = ()

def draw_rectangle(event, x, y, flags, param):
    global start_point, drawing, boxes, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame = clone.copy()
            cv2.rectangle(frame, start_point, (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        boxes.append((start_point, end_point))
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)

print("Draw up to 10 ROIs. Press 'q' to finish.")

while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q") or len(boxes) == 10:
        break

cv2.destroyAllWindows()

# Output and save box coordinates
rois = []
for i, ((x1, y1), (x2, y2)) in enumerate(boxes):
    x_min, y_min = min(x1, x2), min(y1, y2)
    x_max, y_max = max(x1, x2), max(y1, y2)
    print(f"Box {i+1}: ({x_min}, {y_min}) → ({x_max}, {y_max})")
    rois.append({'x1': int(x_min), 'y1': int(y_min), 'x2': int(x_max), 'y2': int(y_max)})

# Save to YAML
roi_config = {'rois': rois}
with open("shared/roi_config.yaml", "w") as f:
    yaml.dump(roi_config, f)

print(f"Saved {len(rois)} ROIs to shared/roi_config.yaml")