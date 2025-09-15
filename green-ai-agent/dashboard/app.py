import streamlit as st

st.set_page_config(page_title="Green AI Agent", layout="wide")
st.title("Green AI Agent")
st.write("Use the sidebar pages: Metrics, Classify, Recent, Analytics.")
st.caption("Backend base URL is read from .streamlit/secrets.toml (API_BASE)")

