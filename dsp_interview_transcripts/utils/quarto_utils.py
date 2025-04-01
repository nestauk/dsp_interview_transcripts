"""
Data viz utils for use in quarto reports.
Could in future be combined with `viz.py` but note that `viz.py` uses Altair whereas these functions use Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


NESTA_COLOURS = [
    "#0000FF",
    "#FDB633",
    "#18A48C",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    # "#FFFFFF",
    "#000000",
]


def load_data_and_centroids(profession):
    data_viz = pd.read_csv(f"{profession}_data_viz.csv")
    # centroids = pd.read_csv(f'{profession}_centroids.csv')

    data_viz["text_wrapped"] = data_viz["context_formatted"].str.wrap(30)
    data_viz["text_wrapped"] = data_viz["text_wrapped"].apply(lambda x: x.replace("\n", "<br>"))

    duplicates = data_viz.duplicated(subset=["x_c", "y_c", "keywords"], keep="first")

    data_viz.loc[duplicates, "keywords"] = ""
    data_viz["keywords"] = data_viz["keywords"].fillna("")

    cluster_names = data_viz["Name"].unique()

    sorted_list = sorted(
        cluster_names, key=lambda x: int(x.split("_")[0])  # Extract the numeric prefix and convert to int
    )

    return sorted_list, data_viz


def plot(sorted_list, data_viz):

    fig = px.scatter(
        data_viz,
        x="x",
        y="y",
        text="keywords",
        color="Name",
        # hover_data=["file_name", "text_wrapped"],
        hover_data={"Name": True, "file_name": True, "text_wrapped": True, "keywords": False, "x": False, "y": False},
        custom_data=["Name", "file_name", "text_wrapped"],
        color_discrete_sequence=NESTA_COLOURS,
        opacity=0.3,
        category_orders={"Name": sorted_list},
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=14, color="black"),  # Increase font size  # Text color
    )

    fig.update_layout(
        width=1200,  # Increase width
        height=800,  # Adjust height
        xaxis=dict(showticklabels=False, title_text=""),  # Hide x-axis ticks and title
        yaxis=dict(showticklabels=False, title_text=""),  # Hide y-axis ticks and title
        legend_title_text="",  # Hide legend title
        plot_bgcolor="white",  # Set background of the plot area to white
        paper_bgcolor="white",  # Set background of the entire figure to white
    )

    return fig


def make_barplots(data_viz, sorted_list):
    row_counts = data_viz["Name"].value_counts().reset_index()
    row_counts.columns = ["Name", "row_count"]
    row_counts["Name"] = pd.Categorical(row_counts["Name"], categories=sorted_list, ordered=True)
    row_counts = row_counts.sort_values("Name")

    # Count of distinct 'file_name' values for each 'Name'
    distinct_file_counts = data_viz.groupby("Name")["file_name"].nunique().reset_index()
    distinct_file_counts.columns = ["Name", "distinct_file_count"]
    distinct_file_counts["Name"] = pd.Categorical(distinct_file_counts["Name"], categories=sorted_list, ordered=True)
    distinct_file_counts = distinct_file_counts.sort_values("Name")

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Number of excerpts", "Number of informants"), shared_yaxes=True
    )

    # Add the first bar plot (row counts)
    fig.add_trace(
        go.Bar(y=row_counts["Name"], x=row_counts["row_count"], orientation="h", name="N excerpts"), row=1, col=1
    )

    # Add the second bar plot (distinct file counts)
    fig.add_trace(
        go.Bar(
            y=distinct_file_counts["Name"],
            x=distinct_file_counts["distinct_file_count"],
            orientation="h",
            name="N informants",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(title_text=None, xaxis_title=None, yaxis_title=None, height=600)

    return fig
