import streamlit as st
import pandas as pd
import numpy as np
import time


st.title("Documentation code dump")

body = """This file shows most of what's in the documentation, use it as a reference, alone
with the [cheatsheet](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py)."""
st.write(body)


st.markdown("## The uber data")

DATE_COLUMN = "date-time"

# @st.cache
@st.cache(suppress_st_warning=True)  # https://docs.streamlit.io/en/stable/caching.html
def load_data(nrows):
    DATA_URL = (
        "https://s3-us-west-2.amazonaws.com/"
        "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    )
    data = (
        pd.read_csv(DATA_URL, nrows=nrows)
        .rename(str.lower, axis="columns")
        .rename(columns={"date/time": "date-time"})
    )
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data["pickup_hour"] = data[DATE_COLUMN].dt.hour
    # NOTE: This will raise a warning, decorate with suppress_st_warning=True to disable it
    st.write("Wuhuu! Got some data!")
    return data


# Create a text element and let the reader kdnow the data is loading.
data_load_state = st.text("Loading data...")
# Load 10,000 rows of data into the dataframe.
data = load_data(nrows=10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...done!")

if st.checkbox("Display the raw data?"):
    st.dataframe(data)

st.markdown("### Number of pickups by hour")

hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)

msg = """
### Map of pickups

Use the sliders to adjust the timespan shown on the map (both inclusive)
"""
st.markdown(msg)

hour_of_day_lower, hour_of_day_upper = st.slider(
    "Span of hours to show", 0, 23, [14, 18]
)
st.write(f"Showing data between {hour_of_day_lower} and {hour_of_day_upper}")

data_mapped = data.query(
    "(pickup_hour >= @hour_of_day_lower) & (pickup_hour <= @hour_of_day_upper)"
)
st.map(data_mapped)
