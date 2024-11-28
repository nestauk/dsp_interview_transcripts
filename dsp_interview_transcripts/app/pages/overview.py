import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input
from dash import Output
from dash import dcc
from dash import html
from data import data_viz
from plotly.subplots import make_subplots


dash.register_page(__name__, path="/overview")

# Sidebar layout
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sentiment_colors = {"Negative": "red", "Neutral": "gray", "Positive": "green"}

# Order topics by prevalence
name_order = data_viz["Name"].value_counts().index.tolist()

# Calculate sentiment percentages
sentiment_percentage = pd.crosstab(data_viz["Name"], data_viz["sentiment"], normalize="index") * 100
sentiment_percentage = sentiment_percentage.reset_index().melt(
    id_vars="Name", var_name="Sentiment", value_name="Percentage"
)

# Combine the charts side-by-side
fig = make_subplots(
    rows=1, cols=2, shared_yaxes=True, subplot_titles=["Prevalence of Each Topic", "Sentiment distribution"]
)

# Add Prevalence chart
fig.add_trace(
    go.Bar(
        x=data_viz["Name"].value_counts().values,
        y=data_viz["Name"].value_counts().index,
        orientation="h",
        name="Prevalence",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Add Sentiment chart
for sentiment in sentiment_percentage["Sentiment"].unique():
    filtered_data = sentiment_percentage[sentiment_percentage["Sentiment"] == sentiment]
    fig.add_trace(
        go.Bar(
            x=filtered_data["Percentage"],
            y=filtered_data["Name"],
            name=sentiment,
            orientation="h",
            marker_color=sentiment_colors[sentiment],
        ),
        row=1,
        col=2,
    )

fig.update_layout(barmode="stack", height=600, yaxis=dict(categoryorder="array", categoryarray=name_order))

layout = html.Div(
    [
        # html.H1("Topic Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Graph(figure=fig)
    ],
    style=CONTENT_STYLE,
)
