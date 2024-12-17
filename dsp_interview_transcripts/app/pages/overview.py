import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input
from dash import Output
from dash import dash_table
from dash import dcc
from dash import html
from data import data_viz
from data import summary_info
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
print(name_order)

# summary_info_sorted = summary_info.set_index("Name").loc[name_order[1:]].reset_index()

# Calculate sentiment percentages
sentiment_percentage = pd.crosstab(data_viz["Name"], data_viz["sentiment"], normalize="index") * 100
sentiment_percentage = sentiment_percentage.reset_index().melt(
    id_vars="Name", var_name="Sentiment", value_name="Percentage"
)

# Calculate number of distinct conversations per topic
distinct_conversations = data_viz.groupby("Name")["conversation"].nunique().reset_index()
distinct_conversations.rename(columns={"conversation": "DistinctConversations"}, inplace=True)


# Combine the charts side-by-side
fig = make_subplots(
    rows=2,
    cols=2,
    shared_yaxes=True,
    subplot_titles=[
        "Number of responses in each topic",
        "Sentiment distribution of responses",
        "Number of users in each topic",
    ],
)

# Add Prevalence chart
fig.add_trace(
    go.Bar(
        x=data_viz["Name"].value_counts().values,
        y=data_viz["Name"].value_counts().index,
        orientation="h",
        name="Prevalence",
        showlegend=False,
        marker_color="#0000FF",
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

# Add Distinct Conversations chart
fig.add_trace(
    go.Bar(
        x=distinct_conversations["DistinctConversations"],
        y=distinct_conversations["Name"],
        orientation="h",
        name="Distinct Conversations",
        marker_color="#FDB633",  # Use a specific color for distinction
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.update_layout(
    barmode="stack",
    height=600,
    yaxis=dict(categoryorder="array", categoryarray=name_order),
    yaxis3=dict(categoryorder="array", categoryarray=name_order),
)

layout = html.Div(
    [
        dcc.Graph(figure=fig),
        html.H4("Topic descriptions and key words", style={"marginTop": "20px"}),
        dash_table.DataTable(
            style_data={"whiteSpace": "normal", "height": "auto"},
            id="summary-table",
            columns=[{"name": col, "id": col} for col in summary_info.columns],
            data=summary_info.to_dict("records"),
            page_action="none",
            style_table={"height": "500px", "overflowY": "auto"},
            style_cell={
                "fontFamily": "Century Gothic",
                "fontSize": "14px",
                "textAlign": "left",
            },
            style_header={"backgroundColor": "#f1f1f1", "fontWeight": "bold", "textAlign": "center"},
            sort_action="native",  # Enable column sorting by the user
        ),
    ],
    style=CONTENT_STYLE,
)
