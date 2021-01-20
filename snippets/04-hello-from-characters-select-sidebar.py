import streamlit as st
from PIL import Image

st.title("Greetings")

greeting_quotes = {
    "Please select a character": None,
    "Gandalf": "You Shall Not Pass!",
    "Deckard Cain": "Hello, my friend. Stay awhile and listen.",
    "King Lonaidas": "This is Sparta!",
}

character = st.sidebar.selectbox("Which character do you like best?", list(greeting_quotes.keys()))

if greeting_quotes[character] is not None:
    msg = f"""
    # _{greeting_quotes[character]}_
    """
    st.markdown(msg)

    img = Image.open(f"../assets/{character}.jpg")
    st.image(img)
