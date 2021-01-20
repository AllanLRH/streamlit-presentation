import streamlit as st

st.title("Greetings")

msg = """
# Greetings (markdown H1)
## Greetings (markdown H2)
### Greetings (markdown H3)
#### Greetings (markdown H4)
##### Greetings (markdown H5)

Well _hello_ to ya'll!

(Streamlit renders output as **markdown** by default.)
"""
st.write(msg)


msg_with_images = R"""
## Hello kitties!

<img src="https://www.rspcasa.org.au/wp-content/uploads/2019/07/kittens-hanging-1-1003x1024.jpg" width="250">

<br>
<br>

![](https://www.rspcasa.org.au/wp-content/uploads/2019/07/kittens-hanging-1-1003x1024.jpg)

"""

st.markdown(msg_with_images, unsafe_allow_html=True)
