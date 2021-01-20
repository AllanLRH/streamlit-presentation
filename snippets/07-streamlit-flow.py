import streamlit as st
from PIL import Image
import requests

# See a list of supported emojis here:
# https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
st.set_page_config(
    page_title=None, page_icon=":dog:", layout="centered", initial_sidebar_state="collapsed"
)

# Download the streamlit logo
r = requests.get("https://docs.streamlit.io/en/stable/_static/logomark_website.png", stream=True)
r.raise_for_status()
img = Image.open(r.raw)

logo_column, header_column = st.beta_columns([4, 20])
with logo_column:
    st.image(img)
with header_column:
    st.title("Streamlit execution flow recap")

body = """
1. Every time a user opens a browser tab pointing to your app, the script is re-executed.
1. Streamlit apps are Python scripts that run from top to bottom.
    - No hidden state, unless you use caching.
    - Clear the cache (press <kbd>C</kbd>) and reload with <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>R</kbd> if caching or state are causing trouble.
1. As the script executes, Streamlit draws its output live in a browser.
1. Scripts use the Streamlit cache to avoid recomputing expensive functions, so updates happen very fast
1. Every time a user interacts with a widget, your script is re-executed and the output value of that widget is set to the new value during that run.
"""

st.markdown(body, unsafe_allow_html=True)


if st.sidebar.button("Dogtax?"):
    img = Image.open("assets/solvej.jpg")
    st.image(img, use_column_width=True)
