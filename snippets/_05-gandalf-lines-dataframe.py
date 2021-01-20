from typing import List

import pandas as pd
import streamlit as st

from gandalf_manus import get_gandalf_lines


# @st.cache
def get_lines(books: List[str]) -> pd.DataFrame:
    to_return = list()
    for book in books:
        lines = get_gandalf_lines(book, delay=0)
        df = pd.DataFrame(lines, columns=["sentences"])
        df["word_count"] = df.sentences.str.split(" ").map(len)
        df["character_count"] = df.sentences.str.replace(" ", "").str.len()
        df["space_count"] = df.sentences.str.count(" ")
        to_return.append(df)
        print(df.shape)
    to_return = pd.DataFrame(to_return) if to_return else pd.DataFrame()
    return to_return.reset_index()


st.sidebar.markdown("Please select the book to grab data from")
keys = list()
if st.sidebar.checkbox("The Fellowship of the Ring"):
    keys.append("The Fellowship of the Ring")
if st.sidebar.checkbox("The Two Towers"):
    keys.append("The Two Towers")
if st.sidebar.checkbox("The Return of the King"):
    keys.append("The Return of the King")

bullets = "".join(["* " + el + "\n" for el in keys])
msg = f"""
# Gandalf lines

Loaded lines for Gandalf for the books:

{bullets}
"""
st.markdown(msg)
df = get_lines(tuple(keys))
st.dataframe(df)
