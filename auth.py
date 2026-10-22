import re
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from cookies import get_cookie, set_cookie

SPREADSHEET_ID = st.secrets.get("spreadsheet_id")
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

def fetch_users_from_db():
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"],
                                                        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet("User Credentials")
    users = sheet.get_all_records()

    if not users:  # No users found
        # Predefine a default admin from secrets
        default_admin = st.secrets["default_admin"]
        if default_admin:
            user_to_db(default_admin['email'], "Yes")
            st.info("Default admin account created. Please restart the app.")
            st.stop()  # Stop execution after adding the admin

    users_dict = {user['email']: user['allow_contact'] for user in users if 'email' in user and 'allow_contact' in user
            }
            
    return users_dict


def login(users):
    if get_cookie("authenticated_email"):
        st.session_state["authenticated"] = True
        st.session_state["email"] = get_cookie("authenticated_email")
        st.session_state["allow_contact"] = get_cookie("allow_contact")
        return True
    
    if not st.session_state.get("authenticated"):
        st.subheader("Login")
        with st.form(key="login_form"):
            email = st.text_input("Please enter your email")
            allow_contact = st.selectbox("Can we contact you for feedback on this tool?", options = ["Yes", "No"])
            submit_button = st.form_submit_button("Login")

        if submit_button:
            email = email.strip()
            if not EMAIL_PATTERN.fullmatch(email):
                st.error("Please enter a valid email address.")
                return False

            if email in users:
                if users[email] != allow_contact:
                    user_to_db(email, allow_contact, update=True)
                    st.session_state["login_message"] = "Your feedback preference has been updated."
                else:
                    st.session_state["login_message"] = "Welcome back!"
            else:
                user_to_db(email, allow_contact)
                st.session_state["login_message"] = "New user registered successfully!"

            st.session_state["authenticated"] = True
            st.session_state["email"] = email
            set_cookie("authenticated_email", email, expiry_days=30)

            # Rerun so the login form is not rendered once the user is authenticated
            st.rerun()
    
    return False
            
def user_to_db(email, allow_contact, update=False):
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"],
                                                        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet("User Credentials")
    
    if update:
        try:
            # Find the row containing the user's email and update the allow_contact column
            email_cell = sheet.find(email)
            sheet.update_cell(email_cell.row, 2, allow_contact)
            return sheet.get_all_records()
        except gspread.exceptions.CellNotFound:
            st.error("Error updating user: Email not found.")
    else:
        sheet.append_row([email, allow_contact])
    

def logout():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.info("You have been logged out.")
        st.rerun()

    