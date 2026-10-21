import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

def save_feedback(email, allow_contact, page, feedback):
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"],
                                                        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(st.secrets.get("spreadsheet_id")).worksheet("Feedback Collection")
    sheet.append_row([email, allow_contact, page, feedback])

def feedback_tool(page):
    with st.form(key='feedback_tool'):

        st.write('Please provide us with your feedback!')
        email = st.text_input('Email')
        feedback = st.text_area('Feedback')
        allow_contact = st.checkbox('I agree to be contacted regarding my feedback')

        submitted = st.form_submit_button('Submit')

        if submitted:
            if feedback and allow_contact:
                if email:
                    save_feedback(email, allow_contact, page, feedback)
                    st.write('Thank you for your feedback!')
                else:
                    st.error('Please provide your email before submitting')
            else:
                save_feedback(email, allow_contact, page, feedback)
                st.write('Thank you for your feedback!')

