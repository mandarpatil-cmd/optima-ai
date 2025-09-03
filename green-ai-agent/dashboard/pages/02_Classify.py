import requests, streamlit as st

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
st.title("🧪 Classify")

text = st.text_area("Text", "Not bad, but not great either.", height=140)
threshold = st.slider("Threshold (light → heavy)", 0.5, 0.999, 0.85, 0.001)
mode = st.selectbox("Mode", ["auto", "light", "heavy"])
force = st.checkbox("Force escalate (debug)", value=False)

if st.button("Classify", use_container_width=True):
    payload = {"text": text, "threshold": float(threshold), "mode": mode, "force_escalate": bool(force)}
    try:
        r = requests.post(f"{API_BASE}/classify", json=payload, timeout=20)
        r.raise_for_status()
        st.success("Done")
        st.json(r.json())
    except Exception as e:
        st.error(f"Call failed: {e}")
