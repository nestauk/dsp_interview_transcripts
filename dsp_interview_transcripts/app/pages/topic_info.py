import dash

from dash import Input
from dash import Output
from dash import dcc
from dash import html
from data import data_viz


dash.register_page(__name__, path="/topic_info")

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

layout = html.Div(
    [
        html.Label("Select a Topic:"),
        dcc.Dropdown(
            id="topic-dropdown",
            options=[{"label": topic, "value": topic} for topic in data_viz["Name"].dropna().unique()],
            placeholder="Choose a topic",
        ),
        html.Div(id="topic-info"),
    ],
    style=CONTENT_STYLE,  # {"padding": "20px"},
)


@dash.callback(Output("topic-info", "children"), Input("topic-dropdown", "value"))
def display_topic_info(selected_topic):
    if selected_topic:
        topic_data = data_viz[data_viz["Name"] == selected_topic]
        n_users = topic_data["conversation"].nunique()
        total_users = data_viz["conversation"].nunique()

        return html.Div(
            [
                html.H4(f"Information about {selected_topic}"),
                html.P(f"Number of mentions: {len(topic_data)}"),
                html.P(f"Number of users in this topic: {n_users} out of {total_users} total users"),
            ]
        )
    return "Select a topic to see more information."
