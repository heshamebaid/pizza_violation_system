import streamlit as st
import requests
import time
from datetime import datetime

st.set_page_config(page_title="Pizza Violation Detection", page_icon="🍕", layout="wide")

st.title("🍕 Pizza Violation Detection System")

frame_url = "http://localhost:8000/frame"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Status & Violations")
    violation_counter_placeholder = st.empty()
    icon_placeholder = st.empty()
    alarm_placeholder = st.empty()
    bbox_placeholder = st.empty()
    label_placeholder = st.empty()
    timestamp_placeholder = st.empty()

last_violation_time = None

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
        violations = data.get('violations', 0)
        violation_counter_placeholder.markdown(f"""
            <div style='display: flex; align-items: center; justify-content: flex-start;'>
                <span style='font-size: 2.5rem; color: red; margin-right: 0.5rem;'>❗</span>
                <span style='font-size: 2.2rem; color: white; background: #d7263d; border-radius: 1.5rem; padding: 0.3rem 1.2rem; font-weight: bold;'>
                    {violations}
                </span>
                <span style='font-size: 1.2rem; color: #d7263d; margin-left: 0.7rem;'>Violations</span>
            </div>
        """, unsafe_allow_html=True)
        if data.get("violation_status"):
            icon_placeholder.markdown("<h1 style='color:red;'>🚨</h1>", unsafe_allow_html=True)
            alarm_placeholder.error("**VIOLATION DETECTED!**")
            if data.get("bboxes"):
                bbox_placeholder.markdown(f"**Bounding Boxes:** `{data['bboxes']}`")
            else:
                bbox_placeholder.empty()
            if data.get("labels"):
                label_placeholder.markdown(f"**Labels:** `{data['labels']}`")
            else:
                label_placeholder.empty()
            # Show timestamp for last violation
            last_violation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamp_placeholder.markdown(f"**Last Violation:** `{last_violation_time}`")
        else:
            icon_placeholder.markdown("<h1 style='color:green;'>✅</h1>", unsafe_allow_html=True)
            alarm_placeholder.success("No violation detected.")
            bbox_placeholder.empty()
            label_placeholder.empty()
            if last_violation_time:
                timestamp_placeholder.markdown(f"**Last Violation:** `{last_violation_time}`")
            else:
                timestamp_placeholder.empty()
    except Exception as e:
        frame_placeholder.write("Could not connect to streaming service.")
        violation_counter_placeholder.empty()
        alarm_placeholder.empty()
        icon_placeholder.empty()
        bbox_placeholder.empty()
        label_placeholder.empty()
        timestamp_placeholder.empty()
    time.sleep(1) 