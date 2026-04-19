import streamlit as st

st.set_page_config(
    page_title="AXON AI",
    page_icon="🧠",
    layout="wide"
)

# Redirect to login page
st.switch_page("pages/1_login.py")