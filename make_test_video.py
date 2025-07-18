import cv2
import numpy as np

# Video properties
width, height = 320, 240
fps = 25
num_seconds = 5
num_frames = fps * num_seconds

# Output path
output_path = "test_output.avi"

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for i in range(num_frames):
    # Create a solid color frame that changes over time
    color = (i % 256, (i*2) % 256, (i*3) % 256)
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    # Overlay frame number
    cv2.putText(frame, f"Frame {i+1}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
    out.write(frame)

out.release()
print(f"Test video saved as: {output_path}") 