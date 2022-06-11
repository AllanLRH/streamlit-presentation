# Same dashboard as before but with logic captured in classes
# Rether than functions. Can aid maintainability quite a bit.

import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


class CachingMessage:
    """
    Handles the caching message displayed in the sidebar.
    """

    def __init__(self) -> None:
        self.msg = """
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

    def render(self) -> None:
        if st.button("Toggle details about caching"):
            st.markdown(self.msg, unsafe_allow_html=True)


class ShowRawData:
    """
    Handles the rendering of the dataframe (the raw data).
    """

    def __init__(self) -> None:
        self.interesting_cols = ["street_nam", "jurisdicti", "installdat", "exisfacil", "shape__length"]

    def _render_interactive_dataframe(self, df: pd.DataFrame):
        # Toggle heatmap on the shape__length column
        toggle_column, _ = st.columns([2, 16])
        with toggle_column:
            heatmap_bool = st.select_slider("Toggle heatmap", ["Off", "On"]) == "On"

        if heatmap_bool:
            format_dict = {"shape__length": "{0:,.1f}"}
            msg = """Pandas' data can be styled when rendering it in a webbrowser; this is a Pandas feature that
                    you can read a blog post about over at [Practical Business Python](https://pbpython.com/styling-pandas.html).
                """
            st.markdown(msg)
            st.dataframe(
                df.style.format(format_dict).background_gradient(subset=["shape__length"], cmap="BuGn")
            )
        else:  # just show the dataframe without highlighting
            st.dataframe(df)

    def _render_static_table(self, df: pd.DataFrame) -> None:
        n_rows = st.slider("How many rows?", 5, 100, 8)
        st.table(df.head(n_rows))

    def render(self, df: pd.DataFrame) -> None:
        show_as = st.selectbox(
            "How should the data be shown?", ["Interactive DataFrame", "Static table"]
        )

        # We're careful not to change the DataFrame from get_data, due to disabling hashing
        df_show = df.loc[:, self.interesting_cols]  # We make a copy of df as to not mutate it

        if show_as == "Interactive DataFrame":
            self._render_interactive_dataframe(df_show)
        elif show_as == "Static table":
            self._render_static_table(df_show)


caching_message = CachingMessage()
show_raw_data = ShowRawData()


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def get_data(delay: int = 7) -> pd.DataFrame:
    """
    Load data, simulate a slow function by sleeping sleep.
    `delay` is in seconds.
    """
    df = (
        pd.read_csv("assets/Existing_Bike_Network.csv")
        .rename(columns=str.lower)
        .set_index("fid")
        .sort_values("installdat")
    )
    df = df.loc[df["installdat"].str.len() == 4, :]
    df["installdat"] = df["installdat"].astype(int)
    time.sleep(delay)  # simulate a slow function
    return df


st.title("Boston bicycle routes (improved caching)")
df = get_data()
# st.write(str(df["installdat"].map(type).value_counts(())))

# This part handles the plotting
st.subheader("Distribution of ride lengths by year")

# These values are needed later
med, std = df.shape__length.median(), df.shape__length.std()

# Let's create two collumns
left_column, right_column = st.columns(2)

n_std = right_column.slider("How many standard deviations from the median should be allowed?", 1, 5, 2)
show_df: bool = left_column.checkbox("Show raw data", False)
grid_bool: bool = left_column.checkbox("Toggle plot grid", True)

dfc = df.query("(shape__length > (@med - @n_std*@std)) & (shape__length < (@med + @n_std*@std))")

fig, ax = plt.subplots(figsize=[8, 4])
if grid_bool:
    ax.grid(linewidth=0.3)
sns.violinplot(
    x=dfc["installdat"],
    y=dfc["shape__length"],
    color="teal",
    linewidth=0,
    ax=ax,
)

st.pyplot(fig)

if show_df:
    show_raw_data.render(df)

# A few notes on caching
with st.sidebar:
    caching_message.render()
