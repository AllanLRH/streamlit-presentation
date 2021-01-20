import streamlit as st
from PIL import Image
import requests
import io

# Download the streamlit logo
r = requests.get("https://docs.streamlit.io/en/stable/_static/logomark_website.png", stream=True)
r.raise_for_status()
img = Image.open(r.raw)

lc, rc = st.beta_columns([4, 20])
lc.image(img)
rc.title("Streamlit execution flow recap")

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
