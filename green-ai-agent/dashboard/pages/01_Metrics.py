import requests, pandas as pd, streamlit as st

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
st.title("📊 Metrics")

@st.cache_data(ttl=5)
def fetch_metrics():
    r = requests.get(f"{API_BASE}/metrics", timeout=10)
    r.raise_for_status()
    return r.json()

try:
    m = fetch_metrics()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Requests", f"{m['requests']:,}")
    c2.metric("Light %", f"{m['light_pct']*100:.1f}%")
    c3.metric("Escalations", f"{m['escalations']:,}")
    c4.metric("Total CO₂ (g)", f"{m['total_co2_g']:.4f}")

    df = pd.DataFrame(m.get("by_model", []))
    if not df.empty:
        st.subheader("By model")
        st.dataframe(df.sort_values("count", ascending=False), use_container_width=True)
        st.subheader("Calls by model")
        st.bar_chart(df.set_index("model")["count"])
    else:
        st.info("No data yet. Make some /classify requests.")
except Exception as e:
    st.error(f"Failed to load metrics from {API_BASE}: {e}")
