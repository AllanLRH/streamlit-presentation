import streamlit as st

body = """
# Streamlit

<img src="https://docs.streamlit.io/en/stable/_static/logomark_website.png">

1. Streamlit apps are Python scripts that run from top to bottom.
1. Every time a user opens a browser tab pointing to your app, the script is re-executed.
1. As the script executes, Streamlit draws its output live in a browser.
1. Scripts use the Streamlit cache to avoid recomputing expensive functions, so updates happen very fast
1. Every time a user interacts with a widget, your script is re-executed and the output value of that widget is set to the new value during that run.
"""

st.markdown(body, unsafe_allow_html=True)
