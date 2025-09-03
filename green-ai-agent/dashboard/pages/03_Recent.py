import requests, pandas as pd, streamlit as st
import os

def get_api_base():
    try:
        return st.secrets["API_BASE"]
    except Exception:
        return os.getenv("API_BASE", "http://127.0.0.1:8000")

API_BASE = get_api_base()
st.title("🧾 Recent Runs")

limit = st.slider("How many rows", 10, 200, 50, 10)

try:
    r = requests.get(f"{API_BASE}/runs/recent", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        st.info("No rows yet. Make some /classify calls.")
    else:
        order = ["id","ts_local","text_hash","mode","label","confidence","model_used","escalated","co2_g"]
        cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
        st.dataframe(df[cols], use_container_width=True)
except Exception as e:
    st.error(f"Failed to load recent runs from {API_BASE}: {e}")
