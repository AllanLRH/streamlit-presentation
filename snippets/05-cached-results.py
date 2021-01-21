import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


@st.cache  # This will cache the output, read more about caching in the docs
def get_data(delay=0):
    df = pd.read_csv("assets/Existing_Bike_Network.csv").rename(columns=str.lower).set_index("fid")
    time.sleep(delay)  # simulate a slow function
    return df


st.title("Boston bicycle routes (caching)")
df = get_data(3).sort_values("installdat")

# This is for toggeling the display of the dataframe...
if st.checkbox("Show raw data", False):
    # ... but let the user choose how to display it...
    show_as = st.selectbox("How should the data be shown?", ["Interactive DataFrame", "Static table"])
    if show_as == "Interactive DataFrame":
        # ... it's interactive, so allow the user to toggle heatmap-highlighting of a row...
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
            # ... or we can just show the dataframe...
            st.dataframe(df)
    elif show_as == "Static table":
        # ... or we can show the data as a "static" table, but since the output
        # can be LARGE, we add a slider to cap the length of the table
        n_rows = st.slider("How many rows?", 5, 100, 8)
        st.table(df.head(n_rows))

# This part handles the plotting
st.subheader("Distribution of ride lengths by year")

n_std = st.slider("How many standard deviations from the median should be allowed?", 1, 10, 10)
grid_bool = st.checkbox("Toggle grid")
med, std = df.shape__length.median(), df.shape__length.std()
# If we have a DataFrame with a lot of rows, filtering will require long (Numpy/Pandas) arrays for
# filtering it, and that can be slow. Thus, we use the query method:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
dfc = df.query("(shape__length > (@med - @n_std*@std)) & (shape__length < (@med + @n_std*@std))")

fig, ax = plt.subplots(figsize=[8, 4])
sns.violinplot(
    x=dfc["installdat"],
    y=dfc["shape__length"],
    color="teal",
    linewidth=0,
    ax=ax,
)
if grid_bool:
    ax.grid()

st.pyplot(fig)
st.pyplot(ax.get_figure())
