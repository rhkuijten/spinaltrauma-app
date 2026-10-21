import streamlit as st
from auth import fetch_users_from_db, logout, login
from menu import create_menu, handle_menu, render_page
from styles import apply_custom_css

apply_custom_css()

def main():
    users = fetch_users_from_db()

    if "cookies" not in st.session_state:
        st.session_state.cookies = {}

    if not st.session_state.get("authenticated"):
        if login(users):
            st.session_state["authenticated"] = True

    if st.session_state.get("authenticated"):
            menu_items = create_menu()
            selected_menu_item = handle_menu(menu_items)
            render_page(selected_menu_item)
            if logout():
                st.rerun()
    else:
        st.info("Please fill in your email address and indicate your feedback preference to continue.")

if __name__ == "__main__":
    main()
