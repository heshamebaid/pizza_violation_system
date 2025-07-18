import streamlit as st
import requests
import time

st.title("Pizza Violation Detection Frontend")

frame_url = "http://localhost:8000/frame"
st.write("Current frame with detections:")
frame_placeholder = st.empty()

st.write("Number of violations (refreshes every 2 seconds):")
violation_placeholder = st.empty()

alarm_placeholder = st.empty()
bbox_placeholder = st.empty()
label_placeholder = st.empty()

while True:
    try:
        # Show current frame
        resp = requests.get(frame_url)
        if resp.status_code == 200:
            frame_placeholder.image(resp.content, channels="BGR")
        else:
            frame_placeholder.write("No frame available.")
        # Show violation count and alarm
        resp2 = requests.get("http://localhost:8000/metadata")
        data = resp2.json()
        violation_placeholder.write(f"Violations: {data.get('violations', 0)}")
        if data.get("violation_status"):
            alarm_placeholder.error("🚨 VIOLATION DETECTED! 🚨")
            if data.get("bboxes"):
                bbox_placeholder.write(f"Bounding Boxes: {data['bboxes']}")
            else:
                bbox_placeholder.empty()
            if data.get("labels"):
                label_placeholder.write(f"Labels: {data['labels']}")
            else:
                label_placeholder.empty()
        else:
            alarm_placeholder.empty()
            bbox_placeholder.empty()
            label_placeholder.empty()
    except Exception as e:
        frame_placeholder.write("Could not connect to streaming service.")
        violation_placeholder.write("")
        alarm_placeholder.empty()
        bbox_placeholder.empty()
        label_placeholder.empty()
    time.sleep(1) 