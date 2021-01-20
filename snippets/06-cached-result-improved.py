import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# NOTE: You may need to clear the cache, see the menu top right, or press 'C' in the browser
@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def get_data(delay=0):
    df = pd.read_csv("assets/Existing_Bike_Network.csv").rename(columns=str.lower).set_index("fid")
    time.sleep(delay)
    return df


st.title("Boston bicycle routes")
df = get_data(3).sort_values("installdat")
if st.checkbox("Show raw data", False):
    show_as = st.selectbox("How should the data be shown?", ["Interactive DataFrame", "Static table"])
    if show_as == "Interactive DataFrame":
        if st.button("Toggle heatmap on `shape__length` column"):
            format_dict = {"shape__length": "{0:,.1f}"}
            msg = """Pandas' data can be styled when rendering it in a webbrowser; this is a Pandas feature that
                     you can read a blog post about over at [Practical Business Python](https://pbpython.com/styling-pandas.html).
                  """
            st.markdown(msg)
            st.dataframe(
                df.style.format(format_dict).background_gradient(subset=["shape__length"], cmap="BuGn")
            )
        else:
            st.dataframe(df)
    elif show_as == "Static table":
        n_rows = st.slider("How many rows?", 5, 100, 8)
        st.table(df.head(n_rows))

st.subheader("Check the boxes to see histograms")

n_std = st.slider("How many standard deviations from the median should be allowed?", 1, 10, 10)
grid_bool = st.checkbox("Toggle grid")
med, std = df.shape__length.median(), df.shape__length.std()
dfc = df.query("(shape__length > (@med - @n_std*@std)) & (shape__length < (@med + @n_std*@std))")

fig, ax = plt.subplots(figsize=[8, 4])
sns.violinplot(
    x=dfc["installdat"], y=dfc["shape__length"], color="teal", linewidth=0, ax=ax,
)
if grid_bool:
    ax.grid()

st.pyplot(ax.get_figure())

if st.button("Toggle details about caching"):
    msg = """
    ### The following elements are tracked for cache invalidation

    1. The input parameters that you called the function with
    1. The value of any external variable used in the function
    1. The body of the function
    1. The body of any function used inside the cached function (but not external libraries)

    ### Good to know about caching

    * A hashmap (dict) is used for caching
    * Caches are stored by reference
    * The cached holds retun-value and a hash of said value

    For more information, read the documentation section<br>
    _[Improve app performance → Advanced Caching](https://docs.streamlit.io/en/stable/caching.html#advanced-caching)_.
    """
    st.markdown(msg, unsafe_allow_html=True)
