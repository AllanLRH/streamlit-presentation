import streamlit as st
from streamlit_vega_lite import vega_lite_component, altair_component
import pydeck as pdk
import graphviz as graphviz
import pandas as pd
import numpy as np
import requests
import plotly.figure_factory as ff
from PIL import Image
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from bokeh.palettes import brewer
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_file, show
from bokeh.models import HoverTool
from bokeh.plotting import figure

st.title("Documentation code dump")

body = """This file shows most of what's in the documentation, use it as a reference, alone
with the [cheatsheet](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py)."""
st.write(body)


st.header("Multi-select menues are available")

options = st.multiselect(
    "What are your favorite colors", ["Green", "Yellow", "Red", "Blue"], ["Yellow", "Red"]
)
st.write("You selected:", options)


st.header("The uber data")

DATE_COLUMN = "date-time"

# @st.cache
@st.cache(suppress_st_warning=True)  # https://docs.streamlit.io/en/stable/caching.html
def load_data(nrows):
    DATA_URL = "https://s3-us-west-2.amazonaws.com/" "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
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

st.subheader("Number of pickups by hour")

hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)


st.subheader("Map of pickups")
st.markdown("Use the sliders to adjust the timespan shown on the map (both inclusive)")

hour_of_day_lower, hour_of_day_upper = st.slider("Span of hours to show", 0, 23, [14, 18])
st.write(f"Showing data between {hour_of_day_lower} and {hour_of_day_upper}")

data_mapped = data.query("(pickup_hour >= @hour_of_day_lower) & (pickup_hour <= @hour_of_day_upper)")
st.map(data_mapped)

st.header("Hope you are into LaTeX?")

ans = st.selectbox("Which kind?", ["Not for me!", "Clothing", "Math"], index=0)
if ans == "Not for me!":
    st.write("Move along, nothing to see here.")
elif ans == "Clothing":
    url = "https://maskinx.com/wp-content/uploads/2019/07/latex_men_jacket_hoodie_maskinx.com01.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    st.image(img, width=400)
elif ans == "Math":
    st.latex(
        r"""
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     """
    )


st.header("JSON can be displayed")

emojis = {
    "curly_loop": "➰",
    "loop": "➿",
    "arrow_heading_up": "⤴️",
    "arrow_heading_down": "⤵️",
    "arrow_left": "⬅️",
    "arrow_up": "⬆️",
    "arrow_down": "⬇️",
    "black_large_square": "⬛",
    "white_large_square": "⬜",
    "star": "⭐",
    "o": "⭕",
    "hankey": "💩",
    "wavy_dash": "〰️",
    "part_alternation_mark": "〽️",
    "congratulations": "㊗️",
    "secret": "㊙️",
}

st.json(emojis)


st.header("Some charts are build in")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.subheader("line_chart")
st.line_chart(chart_data)

st.subheader("area_chart")
st.area_chart(chart_data)

st.subheader("bar_chart")
st.bar_chart(chart_data)


st.header("Pydeck are used for maps, or just called using st.map")


df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"])
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)

st.map(df)

st.header("Graphviz are also easily called")

st.write(
    "You can create the graph using the dot language (Graphviz language) or the objectoriented Python interface"
)

# Create a graphlib graph object
graph = graphviz.Digraph()
graph.edge("run", "intr")
graph.edge("intr", "runbl")
graph.edge("runbl", "run")
graph.edge("run", "kernel")
graph.edge("kernel", "zombie")
graph.edge("kernel", "sleep")
graph.edge("kernel", "runmem")
graph.edge("sleep", "swap")
graph.edge("swap", "runswap")
graph.edge("runswap", "new")
graph.edge("runswap", "runmem")
graph.edge("new", "runmem")
graph.edge("sleep", "runmem")

str_definition_of_graph = """
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
"""

lc, rc = st.beta_columns(2)
lc.graphviz_chart(graph, use_container_width=True)
rc.graphviz_chart(str_definition_of_graph, use_container_width=True)


st.header("We can plot using Altair...")

st.write("Check out the [gallery at the Altair website](https://altair-viz.github.io/gallery/)!")

df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
c = alt.Chart(df).mark_circle().encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
st.altair_chart(c, use_container_width=True)

# Generating Data
source = pd.DataFrame(
    {
        "Trial A": np.random.normal(0, 0.8, 1000),
        "Trial B": np.random.normal(-2, 1, 1000),
        "Trial C": np.random.normal(3, 2, 1000),
    }
)

c = (
    alt.Chart(source)
    .transform_fold(["Trial A", "Trial B", "Trial C"], as_=["Experiment", "Measurement"])
    .mark_area(opacity=0.3, interpolate="step")
    .encode(
        alt.X("Measurement:Q", bin=alt.Bin(maxbins=100)),
        alt.Y("count()", stack=None),
        alt.Color("Experiment:N"),
    )
)

st.altair_chart(c, use_container_width=True)

from vega_datasets import data

counties = alt.topo_feature(data.us_10m.url, "counties")
source = data.unemployment.url
# df = pd.read_csv(source, sep="\t")
# st.write(df)

c = (
    alt.Chart(counties)
    .mark_geoshape()
    .encode(color="rate:Q")
    .transform_lookup(lookup="id", from_=alt.LookupData(source, "id", ["rate"]))
    .project(type="albersUsa")
    .properties(width=500, height=300)
)


st.altair_chart(c, use_container_width=True)


airports = data.airports()
states = alt.topo_feature(data.us_10m.url, feature="states")

# US states background
background = (
    alt.Chart(states)
    .mark_geoshape(fill="lightgray", stroke="white")
    .properties(width=500, height=300)
    .project("albersUsa")
)

# airport positions on background
points = (
    alt.Chart(airports)
    .mark_circle(size=10, color="steelblue")
    .encode(longitude="longitude:Q", latitude="latitude:Q", tooltip=["name", "city", "state"])
)

c = background + points
st.altair_chart(c, use_container_width=True)

source = data.movies.url

c = (
    alt.Chart(source)
    .mark_rect()
    .encode(
        alt.X("IMDB_Rating:Q", bin=alt.Bin(maxbins=60)),
        alt.Y("Rotten_Tomatoes_Rating:Q", bin=alt.Bin(maxbins=40)),
        alt.Color("count(IMDB_Rating):Q", scale=alt.Scale(scheme="cividis")),
    )
)
st.altair_chart(c, use_container_width=True)


source = data.seattle_weather.url

step = 20
overlap = 1

c = (
    alt.Chart(source, height=step)
    .transform_timeunit(Month="month(date)")
    .transform_joinaggregate(mean_temp="mean(temp_max)", groupby=["Month"])
    .transform_bin(["bin_max", "bin_min"], "temp_max")
    .transform_aggregate(value="count()", groupby=["Month", "mean_temp", "bin_min", "bin_max"])
    .transform_impute(impute="value", groupby=["Month", "mean_temp"], key="bin_min", value=0)
    .mark_area(interpolate="monotone", fillOpacity=0.8, stroke="lightgray", strokeWidth=0.5)
    .encode(
        alt.X("bin_min:Q", bin="binned", title="Maximum Daily Temperature (C)"),
        alt.Y("value:Q", scale=alt.Scale(range=[step, -step * overlap]), axis=None),
        alt.Fill("mean_temp:Q", legend=None, scale=alt.Scale(domain=[30, 5], scheme="redyellowblue")),
    )
    .facet(
        row=alt.Row(
            "Month:T", title=None, header=alt.Header(labelAngle=0, labelAlign="right", format="%B")
        )
    )
    .properties(title="Seattle Weather", bounds="flush")
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
    .configure_title(anchor="end")
)

st.altair_chart(c, use_container_width=True)

source = data.iris()

c = (
    alt.Chart(source)
    .transform_window(index="count()")
    .transform_fold(["petalLength", "petalWidth", "sepalLength", "sepalWidth"])
    .mark_line()
    .encode(x="key:N", y="value:Q", color="species:N", detail="index:N", opacity=alt.value(0.5))
    .properties(width=500)
)

st.altair_chart(c, use_container_width=True)


source = data.iris()

c = (
    alt.Chart(source)
    .transform_window(index="count()")
    .transform_fold(["petalLength", "petalWidth", "sepalLength", "sepalWidth"])
    .transform_joinaggregate(min="min(value)", max="max(value)", groupby=["key"])
    .transform_calculate(
        minmax_value=(alt.datum.value - alt.datum.min) / (alt.datum.max - alt.datum.min),
        mid=(alt.datum.min + alt.datum.max) / 2,
    )
    .mark_line()
    .encode(x="key:N", y="minmax_value:Q", color="species:N", detail="index:N", opacity=alt.value(0.5))
    .properties(width=500)
)
st.altair_chart(c, use_container_width=True)


source = data.movies.url

pts = alt.selection(type="single", encodings=["x"])

rect = (
    alt.Chart(data.movies.url)
    .mark_rect()
    .encode(
        alt.X("IMDB_Rating:Q", bin=True),
        alt.Y("Rotten_Tomatoes_Rating:Q", bin=True),
        alt.Color(
            "count()", scale=alt.Scale(scheme="greenblue"), legend=alt.Legend(title="Total Records")
        ),
    )
)

circ = (
    rect.mark_point()
    .encode(alt.ColorValue("grey"), alt.Size("count()", legend=alt.Legend(title="Records in Selection")))
    .transform_filter(pts)
)

bar = (
    alt.Chart(source)
    .mark_bar()
    .encode(
        x="Major_Genre:N",
        y="count()",
        color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey")),
    )
    .properties(width=550, height=200)
    .add_selection(pts)
)

c = alt.vconcat(rect + circ, bar).resolve_legend(color="independent", size="independent")

st.altair_chart(c, use_container_width=True)


source = data.unemployment_across_industries.url

selection = alt.selection_multi(fields=["series"], bind="legend")

c = (
    alt.Chart(source)
    .mark_area()
    .encode(
        alt.X("yearmonth(date):T", axis=alt.Axis(domain=False, format="%Y", tickSize=0)),
        alt.Y("sum(count):Q", stack="center", axis=None),
        alt.Color("series:N", scale=alt.Scale(scheme="category20b")),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    )
    .add_selection(selection)
)


st.altair_chart(c, use_container_width=True)


st.header("... or Plotly ...")

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
# Group data together
hist_data = [x1, x2, x3]
group_labels = ["Group 1", "Group 2", "Group 3"]
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])
# Plot!
st.plotly_chart(fig, use_container_width=True)


mesh_size = 0.02
margin = 1

# We will use the iris data, which is included in px
df = px.data.iris()
df_train, df_test = train_test_split(df, test_size=0.25, random_state=0)
X_train = df_train[["sepal_length", "sepal_width"]]
y_train = df_train.species_id

# Create a mesh grid on which we will run our model
l_min, l_max = df.sepal_length.min() - margin, df.sepal_length.max() + margin
w_min, w_max = df.sepal_width.min() - margin, df.sepal_width.max() + margin
lrange = np.arange(l_min, l_max, mesh_size)
wrange = np.arange(w_min, w_max, mesh_size)
ll, ww = np.meshgrid(lrange, wrange)

# Create classifier, run predictions on grid
clf = KNeighborsClassifier(15, weights="distance")
clf.fit(X_train, y_train)
Z = clf.predict(np.c_[ll.ravel(), ww.ravel()])
Z = Z.reshape(ll.shape)
proba = clf.predict_proba(np.c_[ll.ravel(), ww.ravel()])
proba = proba.reshape(ll.shape + (3,))

# Compute the confidence, which is the difference
diff = proba.max(axis=-1) - (proba.sum(axis=-1) - proba.max(axis=-1))

fig = px.scatter(
    df_test,
    x="sepal_length",
    y="sepal_width",
    symbol="species",
    symbol_map={"setosa": "square-dot", "versicolor": "circle-dot", "virginica": "diamond-dot"},
)
fig.update_traces(marker_size=12, marker_line_width=1.5, marker_color="lightyellow")
fig.add_trace(
    go.Heatmap(
        x=lrange,
        y=wrange,
        z=diff,
        opacity=0.25,
        customdata=proba,
        colorscale="RdBu",
        hovertemplate=(
            "sepal length: %{x} <br>"
            "sepal width: %{y} <br>"
            "p(setosa): %{customdata[0]:.3f}<br>"
            "p(versicolor): %{customdata[1]:.3f}<br>"
            "p(virginica): %{customdata[2]:.3f}<extra></extra>"
        ),
    )
)
fig.update_layout(legend_orientation="h", title="Prediction Confidence on Test Split")
st.plotly_chart(fig, use_container_width=True)

st.header("... or Bokeh")

N = 10
df = pd.DataFrame(np.random.randint(10, 100, size=(15, N))).add_prefix("y")

p = figure(x_range=(0, len(df) - 1), y_range=(0, 800))
p.grid.minor_grid_line_color = "#eeeeee"

names = ["y%d" % i for i in range(N)]
p.varea_stack(stackers=names, x="index", color=brewer["Spectral"][N], legend_label=names, source=df)

# reverse the legend entries to match the stacked order
p.legend.items.reverse()

st.bokeh_chart(p, use_container_width=True)


n = 500
x = 2 + 2 * np.random.standard_normal(n)
y = 2 + 2 * np.random.standard_normal(n)

p = figure(
    title="Hexbin for 500 points",
    match_aspect=True,
    tools="wheel_zoom,reset",
    background_fill_color="#440154",
)
p.grid.visible = False

r, bins = p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)

p.circle(x, y, color="white", size=1)

p.add_tools(
    HoverTool(
        tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)")],
        mode="mouse",
        point_policy="follow_mouse",
        renderers=[r],
    )
)


st.bokeh_chart(p, use_container_width=True)
