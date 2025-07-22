import cv2
import yaml
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog

class ROISelector:
    def __init__(self, video_path, config_path="shared/roi_config.yaml"):
        self.video_path = video_path
        self.config_path = config_path
        self.boxes = []
        self.roi_names = []
        self.drawing = False
        self.start_point = ()
        self.current_roi_name = ""
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 255, 0)
        ]
        
        self.load_existing_rois()
        self.load_frame()
        
    def load_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Could not read frame from video.")
            exit(1)
            
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.clone = frame.copy()
        
        print(f"Frame loaded: {frame.shape[1]}x{frame.shape[0]}")
        
    def load_existing_rois(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    existing_rois = config.get('rois', [])
                for i, roi in enumerate(existing_rois):
                    start = (roi['x1'], roi['y1'])
                    end = (roi['x2'], roi['y2'])
                    self.boxes.append((start, end))
                    self.roi_names.append(roi.get('name', f'ROI_{i+1}'))
                print(f"Loaded {len(self.boxes)} existing ROIs")
            except Exception as e:
                print(f"Could not load existing ROIs: {e}")
                
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.frame = self.clone.copy()
                self.draw_existing_boxes()
                cv2.rectangle(self.frame, self.start_point, (x, y), (0, 255, 255), 2)
                coord_text = f"({self.start_point[0]}, {self.start_point[1]}) -> ({x}, {y})"
                cv2.putText(self.frame, coord_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            if abs(end_point[0] - self.start_point[0]) > 10 and abs(end_point[1] - self.start_point[1]) > 10:
                self.boxes.append((self.start_point, end_point))
                roi_name = self.get_roi_name(len(self.boxes))
                self.roi_names.append(roi_name)
                print(f"Added ROI {len(self.boxes)}: {roi_name}")
                self.update_display()
            else:
                print("ROI too small, minimum size is 10x10 pixels")
                self.frame = self.clone.copy()
                self.draw_existing_boxes()
                
    def get_roi_name(self, roi_number):
        """Prompt user for custom ROI name using tkinter popup"""
        try:
            root = tk.Tk()
            root.withdraw()
            name = simpledialog.askstring(title="ROI Naming",
                                          prompt=f"Enter name for ROI {roi_number}:")
            root.destroy()
            if name:
                return name.strip()
        except Exception as e:
            print(f"Error during name input: {e}")
        return f"ROI_{roi_number}"
        
    def draw_existing_boxes(self):
        for i, ((x1, y1), (x2, y2)) in enumerate(self.boxes):
            color = self.colors[i % len(self.colors)]
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
            roi_name = self.roi_names[i] if i < len(self.roi_names) else f"ROI_{i+1}"
            label = f"{i+1}: {roi_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(self.frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(self.frame, label, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
    def update_display(self):
        self.frame = self.clone.copy()
        self.draw_existing_boxes()
        instructions = [
            f"ROIs: {len(self.boxes)}/10",
            "Left click & drag: Draw ROI",
            "'r': Remove last ROI",
            "'c': Clear all ROIs", 
            "'s': Save & exit",
            "'q': Exit without saving"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(self.frame, instruction, (10, self.frame.shape[0] - 150 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
    def remove_last_roi(self):
        if self.boxes:
            removed_roi = self.roi_names.pop() if self.roi_names else f"ROI_{len(self.boxes)}"
            self.boxes.pop()
            print(f"Removed: {removed_roi}")
            self.update_display()
        else:
            print("No ROIs to remove")
            
    def clear_all_rois(self):
        if self.boxes:
            self.boxes.clear()
            self.roi_names.clear()
            print("Cleared all ROIs")
            self.update_display()
        else:
            print("No ROIs to clear")
            
    def save_rois(self):
        if not self.boxes:
            print("No ROIs to save")
            return False
        rois = []
        for i, ((x1, y1), (x2, y2)) in enumerate(self.boxes):
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            roi_name = self.roi_names[i] if i < len(self.roi_names) else f"ROI_{i+1}"
            roi_data = {
                'name': roi_name,
                'x1': int(x_min), 'y1': int(y_min), 
                'x2': int(x_max), 'y2': int(y_max),
                'area': (x_max - x_min) * (y_max - y_min),
                'center_x': int((x_min + x_max) / 2),
                'center_y': int((y_min + y_max) / 2)
            }
            rois.append(roi_data)
            print(f"ROI {i+1} ({roi_name}): ({x_min}, {y_min}) → ({x_max}, {y_max}) [Area: {roi_data['area']}px²]")
        roi_config = {
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'video_source': os.path.basename(self.video_path),
                'frame_resolution': f"{self.original_frame.shape[1]}x{self.original_frame.shape[0]}",
                'total_rois': len(rois)
            },
            'rois': rois
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(roi_config, f, default_flow_style=False, sort_keys=False)
        ref_image_path = self.config_path.replace('.yaml', '_reference.jpg')
        cv2.imwrite(ref_image_path, self.frame)
        print(f"\n✓ Saved {len(rois)} ROIs to {self.config_path}")
        print(f"✓ Saved reference image to {ref_image_path}")
        return True
        
    def run(self):
        print("=" * 60)
        print("Enhanced ROI Selection Tool")
        print("=" * 60)
        print(f"Video: {os.path.basename(self.video_path)}")
        print(f"Config: {self.config_path}")
        print(f"Existing ROIs: {len(self.boxes)}")
        print("=" * 60)
        cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Selector", 1200, 800)
        cv2.setMouseCallback("ROI Selector", self.draw_rectangle)
        self.update_display()
        print("\nInstructions:")
        print("- Left click & drag to draw ROI rectangles")
        print("- 'r' key: Remove last ROI")
        print("- 'c' key: Clear all ROIs")
        print("- 's' key: Save ROIs and exit")
        print("- 'q' key: Exit without saving")
        print(f"- Draw up to 10 ROIs for ingredient areas")
        while True:
            cv2.imshow("ROI Selector", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nExiting without saving...")
                break
            elif key == ord("s"):
                if self.save_rois():
                    break
            elif key == ord("r"):
                self.remove_last_roi()
            elif key == ord("c"):
                self.clear_all_rois()
            elif len(self.boxes) >= 10:
                print(f"\nMaximum of 10 ROIs reached. Press 's' to save or 'r' to remove some.")
        cv2.destroyAllWindows()

def main():
    video_path = "shared/videos/Sah w b3dha ghalt (2).mp4"
    config_path = "shared/roi_config.yaml"
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    selector = ROISelector(video_path, config_path)
    selector.run()

if __name__ == "__main__":
    main()
