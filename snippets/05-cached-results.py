import streamlit as st
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache
def get_data(delay=0):
    df = (
        pd.read_csv("assets/Existing_Bike_Network.csv")
        .rename(columns=str.lower)
        .set_index("fid")
    )
    time.sleep(delay)
    return df


st.title("Boston bicycle routes")
df = get_data(3).sort_values("installdat")
if st.checkbox("Show raw data", False):
    show_as = st.selectbox(
        "How should the data be shown?", ["Interactive DataFrame", "Static table"]
    )
    if show_as == "Interactive DataFrame":
        st.dataframe(df)
    elif show_as == "Static table":
        n_rows = st.slider("How many rows?", 5, 100, 8)
        st.table(df.head(n_rows))

st.subheader("Check the boxes to see histograms")

n_std = st.slider(
    "How many standard deviations from the median should be allowed?", 1, 10, 10
)
med, std = df.shape__length.median(), df.shape__length.std()
dfc = df.query(
    f"(shape__length > (@med - {n_std}*@std)) & (shape__length < (@med + {n_std}*@std))"
)

fig, ax = plt.subplots(figsize=[8, 4])
sns.violinplot(
    x=dfc["installdat"],
    y=dfc["shape__length"],
    color="teal",
    linewidth=0,
    ax=ax,
)
st.pyplot(ax.get_figure())
