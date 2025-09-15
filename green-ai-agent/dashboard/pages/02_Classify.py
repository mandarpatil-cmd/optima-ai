import requests
import streamlit as st

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
st.title("Classify")

text = st.text_area("Text", "Not bad, but not great either.", height=140)
threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.85, 0.01)
priority = st.selectbox("Priority", ["balanced", "accuracy", "speed", "energy"], index=0)

if st.button("Classify", use_container_width=True):
    payload = {
        "text": text,
        "preferences": {"confidence_threshold": float(threshold), "priority": priority},
    }
    try:
        r = requests.post(f"{API_BASE}/classify-email", json=payload, timeout=20)
        r.raise_for_status()
        st.success("Done")
        st.json(r.json())
    except Exception as e:
        st.error(f"Call failed: {e}")

