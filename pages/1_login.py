import streamlit as st
st.set_page_config(page_title="AXON · Sign In", layout="centered")

from utils import inject_css, init_session
from modules.auth_service import login_user, register_user

inject_css()
init_session()

# Redirect if already logged in
if st.session_state.get("user_id"):
    st.switch_page("pages/3_upload.py")

# Default mode
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# ── UI HEADER ──
st.markdown("""
<div style="text-align:center;padding:40px 0 28px;">
  <div style="font-size:26px;font-weight:800;">AXON</div>
  <div style="font-size:12px;color:gray;">AI Data Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

# ── CARD ──
_, col, _ = st.columns([1, 2, 1])

with col:
    if st.session_state.auth_mode == "login":

        st.subheader("Login")

        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            uid, err = login_user(email, password)

            if uid:
                st.session_state.user_id = uid
                st.session_state.username = email
                st.switch_page("pages/3_upload.py")
            else:
                st.error(err)

        st.write("Don't have an account?")
        if st.button("Create Account"):
            st.session_state.auth_mode = "register"
            st.rerun()

    else:
        st.subheader("Register")

        with st.form("register_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")

        if submitted:
            if password != confirm:
                st.error("Passwords do not match")
            else:
                ok, msg = register_user(email, password)

                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        if st.button("Back to Login"):
            st.session_state.auth_mode = "login"
            st.rerun()