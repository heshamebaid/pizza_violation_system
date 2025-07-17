import streamlit as st
import requests
import time

st.title("Pizza Violation Detection Frontend")

frame_url = "http://localhost:8000/frame"
st.write("Current frame with detections:")
frame_placeholder = st.empty()

st.write("Number of violations (refreshes every 2 seconds):")
violation_placeholder = st.empty()

while True:
    try:
        # Show current frame
        resp = requests.get(frame_url)
        if resp.status_code == 200:
            frame_placeholder.image(resp.content, channels="BGR")
        else:
            frame_placeholder.write("No frame available.")
        # Show violation count
        resp2 = requests.get("http://localhost:8000/metadata")
        data = resp2.json()
        violation_placeholder.write(f"Violations: {data.get('violations', 0)}")
    except Exception as e:
        frame_placeholder.write("Could not connect to streaming service.")
        violation_placeholder.write("")
    time.sleep(1) 