# We can use markdown, and even render HTML

import streamlit as st

# This is the streamlit way to set a title...
st.title("Greetings")

# But we can also use headers from markdown
msg = """
# Greetings (markdown H1)
## Greetings (markdown H2)
### Greetings (markdown H3)
#### Greetings (markdown H4)
##### Greetings (markdown H5)

Well _hello_ to ya'll!

(Streamlit renders output as **markdown** by default.)
"""
# Streamlit will guess that it's markdown...
st.write(msg)


msg_with_images = R"""
## Hello kitties!

<img src="https://www.rspcasa.org.au/wp-content/uploads/2019/07/kittens-hanging-1-1003x1024.jpg" width="250">

<br>
<br>

![](https://www.rspcasa.org.au/wp-content/uploads/2019/07/kittens-hanging-1-1003x1024.jpg)

"""
# ...but if we want to do "advanced" markdown stuff, like including images, we need the st.markdown function
st.markdown(msg_with_images, unsafe_allow_html=True)
