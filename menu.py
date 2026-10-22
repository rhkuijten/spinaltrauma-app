import streamlit as st
import streamlit_antd_components as sac
from Pages.Trauma import trauma, trauma_home
from Pages.Trauma import evaluation

# SCAFFOLD: point this at the trauma manuscript once it is published.
MANUSCRIPT_URL = 'https://pubmed.ncbi.nlm.nih.gov'
GITHUB_URL = 'https://github.com/rhkuijten/spine-trauma-30day-mortality'

def create_menu():
    menu_items = [
        sac.MenuItem('Home', icon='house-fill'),
        sac.MenuItem('Decision tool', icon='speedometer2'),
        sac.MenuItem('Evaluation', icon='clipboard-data'),
        sac.MenuItem('Links', type='group', children=[
                sac.MenuItem('Manuscript & Supplements', icon='file-earmark-text', href=MANUSCRIPT_URL),
                sac.MenuItem('Github Source Code', icon='github', href=GITHUB_URL)
            ])
    ]
    return menu_items

def handle_menu(menu_items):
    with st.sidebar:
        selected_menu_item = sac.menu(menu_items, open_all=True)
    return selected_menu_item

def render_page(selected_menu_item):
    if selected_menu_item == 'Home':
        trauma_home.trauma_home_page()
    elif selected_menu_item == 'Decision tool':
        trauma.trauma_page()
    elif selected_menu_item == 'Evaluation':
        evaluation.eval_page()
