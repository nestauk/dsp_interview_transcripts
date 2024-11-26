import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from dash import Input
from dash import Output
from dash import dash_table
from dash import dcc
from dash import html

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts.getters.interim import get_data_w_topics
from dsp_interview_transcripts.getters.interim import get_rep_docs
from dsp_interview_transcripts.getters.interim import get_topic_names
from dsp_interview_transcripts.getters.raw import get_raw_transcripts_cleaned


rep_docs = get_rep_docs(production=True)
data = get_data_w_topics(production=True)
data_w_names = get_topic_names(production=True)

topic_counts = pd.DataFrame(data["Cluster"].value_counts()).reset_index()
topic_counts = topic_counts.rename(columns={"count": "N responses in topic"})

data_w_names = data_w_names.rename(columns={"llama3.2_name": "Name", "llama3.2_description": "Description"})
data_w_names = pd.merge(data_w_names, topic_counts, left_on="Cluster", right_on="Cluster", how="left")

data_viz = (
    data.merge(data_w_names[["Cluster", "Name", "Description"]], on="Cluster", how="left")
    .assign(Name=lambda df: df["Name"].fillna("None"))
    .assign(Description=lambda df: df["Description"].fillna("None"))
)

transcripts = get_raw_transcripts_cleaned()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("QualFML", className="display-5"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Information by topic", href="/page-1", active="exact"),
                dbc.NavLink("User response mapping", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("Welcome!")
    elif pathname == "/page-1":
        # return html.P("This is the content of page 1. Yay!")
        return html.Div(
            [
                html.Label("Select a Topic:"),
                dcc.Dropdown(
                    id="topic-dropdown",
                    options=[
                        {"label": topic, "value": topic}
                        for topic in data_viz["Name"]
                        .dropna()
                        .unique()  # This is necessary because Dash can't handle "None" for some reason
                    ],
                    placeholder="Choose a topic",
                ),
                html.Div(id="topic-info"),
            ],
            style={"padding": "20px"},
        )
    elif pathname == "/page-2":
        return html.Div(
            [
                # scatterplot
                html.Div(
                    [
                        dcc.Graph(id="scatter-plot"),
                    ],
                    style={"width": "100%", "marginBottom": "20px"},
                ),
                # table
                html.Div(
                    [
                        dash_table.DataTable(
                            style_data={
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            id="filtered-table",
                            columns=[{"name": i, "id": i} for i in ["uuid", "role", "text"]],
                            data=[],
                            style_data_conditional=[],
                            page_action="none",
                            style_table={"height": "500px", "overflowY": "auto"},
                        )
                    ],
                    style={
                        "width": "100%",
                    },
                ),
            ]
        )
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(Output("topic-info", "children"), Input("topic-dropdown", "value"))
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


# Callback to update scatter plot
@app.callback(Output("scatter-plot", "figure"), Input("scatter-plot", "id"))
def update_scatter_plot(_):
    fig = px.scatter(
        data_viz,
        x="x",
        y="y",
        color="Name",
        hover_data=["conversation", "text_clean"],
        custom_data=["conversation", "text_clean", "uuid"],
    )
    fig.update_layout(transition_duration=500)
    return fig


# Callback to update table based on scatter plot selection
@app.callback(
    [Output("filtered-table", "data"), Output("filtered-table", "style_data_conditional")],
    Input("scatter-plot", "clickData"),
)
def display_click_data(clickData):
    if clickData:
        # print(clickData['points'][0])
        selected_uuid = clickData["points"][0]["customdata"][2]
        print(selected_uuid)
        conversation_id = clickData["points"][0]["customdata"][0]  # Get the 'conversation' from hover data
        filtered_data = transcripts[transcripts["conversation"] == conversation_id]
        table_data = filtered_data[["uuid", "role", "text"]].to_dict("records")
        style_data_conditional = [
            {
                "if": {"filter_query": f'{{uuid}} = "{selected_uuid}"'},
                "backgroundColor": "#FFDDC1",  # Choose a color for highlighting
                "fontWeight": "bold",
            }
        ]
        return table_data, style_data_conditional
    return [], []


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
