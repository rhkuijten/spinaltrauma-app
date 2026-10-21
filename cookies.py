import streamlit as st
import datetime

def set_cookie(name, value, expiry_days):
    """Set a cookie with an expiration date."""
    expiry = datetime.datetime.now() + datetime.timedelta(days=expiry_days)
    st.session_state.cookies[name] = {"value": value, "expiry": expiry.isoformat()}

def get_cookie(name):
    """Retrieve a cookie if it exists and is not expired."""
    cookie = st.session_state.cookies.get(name)
    if cookie:
        expiry = datetime.datetime.fromisoformat(cookie["expiry"])
        if expiry > datetime.datetime.now():
            return cookie["value"]
    return None

def clear_cookies():
    """Clear all cookies."""
    st.session_state.cookies.clear()
