# We can also use the Pillow library (impoted as PIL) to show images.
# Interactivity is added using a st.selectbox.
# Each time a widget state is changed, the file is executed all over again, but
# with the widget state preserved. The output of the widgets may be used to change
# what's being computed and rendered.


import streamlit as st
from PIL import Image  # use the Pillow library, as PIL is unmaintained

st.title("Greetings")

greeting_quotes = {
    "Please select a character": None,
    "Gandalf": "You Shall Not Pass!",
    "Deckard Cain": "Hello, my friend. Stay awhile and listen.",
    "King Lonaidas": "This is Sparta!",
}

character = st.selectbox("Which character do you like best?", list(greeting_quotes.keys()))

if greeting_quotes[character] is not None:
    msg = f"""
    # _{greeting_quotes[character]}_
    """
    st.markdown(msg)

    img = Image.open(f"assets/{character}.jpg")
    st.image(img)
