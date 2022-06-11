# Demonstrate that Streamlit will render objects that's just "returned"
# This is probably not a good idea compared to st.write and similar, but
# it works in a jiff.
#
# Notice that elements are rendered in the order of evaluation (line-by-line top down, of course).


import pandas as pd

"""
# This is the document title

This is some _markdown_.
"""

df = pd.DataFrame({"col1": [1, 2, 3]})
df  # <-- Draw the dataframe

x = 10
"x", x  # <-- Draw the string 'x' and then the value of x
