import time

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple
import pandas as pd
import seaborn as sns
import streamlit as st


# NOTE: You may need to clear the cache, see the menu top right, or press 'C' in the browser
# This turns off hashing for this DataFrame, because I KNOW that it's not changing
#
# Hashing None is _fast_, hashing a large DataFrame is _slow_.
# Notice that we provide Streamlit with a value to use in it's internal hash function,
# NOT the hashed value of the dataframe.
#
# The hash functinality can be further overwritten to include e.g. contents or filesystem-timestamps
# of a file:
# https://docs.streamlit.io/en/stable/api.html#streamlit.image
@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def get_data(delay=0):
    df = pd.read_csv("assets/Existing_Bike_Network.csv").rename(columns=str.lower).set_index("fid")
    time.sleep(delay)
    return df


@st.cache(hash_funcs={mpl.figure.Figure: lambda _: None})
def get_plot(data: pd.DataFrame, x: str, y: str, grid: bool) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=[8, 4])
    sns.violinplot(data=data, x=x, y=y, color="teal", linewidth=0, ax=ax)
    if grid_bool:
        ax.grid()
    return fig, ax 


# This is a big chunk of code, so it's a good candidate for a function
def show_raw_data():
    show_as = st.selectbox("How should the data be shown?", ["Interactive DataFrame", "Static table"])
    interesting_cols = ["street_nam", "jurisdicti", "installdat", "exisfacil", "shape__length"]
    # We're careful not to change the DataFrame from get_data, due to disabling hashing
    df_show = df.loc[:, interesting_cols]  # We make a copy of df as to not mutate it
    if show_as == "Interactive DataFrame":
        if st.button("Toggle heatmap on `shape__length` column"):
            format_dict = {"shape__length": "{0:,.1f}"}
            msg = """Pandas' data can be styled when rendering it in a webbrowser; this is a Pandas feature that
                     you can read a blog post about over at [Practical Business Python](https://pbpython.com/styling-pandas.html).
                  """
            st.markdown(msg)
            st.dataframe(
                df_show.style.format(format_dict).background_gradient(
                    subset=["shape__length"], cmap="BuGn"
                )
            )
        else:
            st.dataframe(df_show)
    elif show_as == "Static table":
        n_rows = st.slider("How many rows?", 5, 100, 8)
        st.table(df_show.head(n_rows))


st.title("Boston bicycle routes (improved caching)")
df = get_data(delay=3).sort_values("installdat")

# This part handles the plotting
st.subheader("Distribution of ride lengths by year")

# These values are needed later
med, std = df.shape__length.median(), df.shape__length.std()

# Let's create two collumns
left_column, right_column = st.beta_columns(2)

n_std = right_column.slider(
    "How many standard deviations from the median should be allowed?", 0.1, 10.0, 3.5
)
show_df: bool = left_column.checkbox("Show raw data", False)
grid_bool: bool = left_column.checkbox("Toggle plot grid", True)

dfc = df.query("(shape__length > (@med - @n_std*@std)) & (shape__length < (@med + @n_std*@std))")


fig, ax = get_plot(dfc, x="installdat", y="shape__length", grid=True)
st.pyplot(fig)

if show_df:
    show_raw_data()

# A few notes on caching
if st.sidebar.button("Toggle details about caching"):
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
    st.sidebar.markdown(msg, unsafe_allow_html=True)
