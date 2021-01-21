import streamlit as st

# The PIL project is abandonned, but the Pillow package is a drop-in replacement
from PIL import Image


st.title("Greetings")

# Note that this example relies on files from the assets-folder, and relies on the filenames
# exactly matching the keys in the dictionary below.
greeting_quotes = {
    "Please select a character": None,
    "Gandalf": "You Shall Not Pass!",
    "Deckard Cain": "Hello, my friend. Stay awhile and listen.",
    "King Lonaidas": "This is Sparta!",
}

# This renderes the drop-down menu, and `character` will take on the selected value from the dropdown list.
# The initial value is the top item, because we didn't specify anything else.
character = st.sidebar.selectbox("Which character do you like best?", list(greeting_quotes.keys()))

if greeting_quotes[character] is not None:
    msg = f"""
    # _{greeting_quotes[character]}_
    """
    st.markdown(msg)

    # This is the official way of including images.
    # The width can also be set to adaptively fit the width of the
    # display-frame using the argument use_column_width=True
    #
    # The docs are _really_ good, check them out!
    # https://docs.streamlit.io/en/stable/api.html#streamlit.image
    img = Image.open(f"assets/{character}.jpg")
    st.image(img, width=800)
