import streamlit as st

def apply_custom_css():
    max_width_str = """
    <style>
        .block-container {
            max-width: 1100px;
            margin: 0 auto;
            /* The top padding keeps content clear of Streamlit's fixed 3.75rem
               toolbar, which otherwise covers the first heading on each page. */
            padding: 5rem 20px 20px;
        }
        .centered-header {
            text-align: center;
        }
        .section-header {
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .disclaimer {
            color: red;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(max_width_str, unsafe_allow_html=True)


def show_image_or_placeholder(path, label):
    """Render an image if it exists, otherwise show a placeholder box.

    Used by evaluation pages that are still awaiting their figures.
    """
    import os

    if os.path.exists(path):
        st.image(path, use_column_width=True)
    else:
        st.markdown(
            f"""
            <div style="border: 2px dashed rgba(128,128,128,0.5); border-radius: 10px;
                        padding: 48px 16px; text-align: center; color: rgba(128,128,128,0.9);">
                <strong>{label}</strong><br>
                <span style="font-size: 13px;">Figure not yet added &mdash; place it at <code>{path}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
