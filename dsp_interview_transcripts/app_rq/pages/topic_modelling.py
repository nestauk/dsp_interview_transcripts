import dash
import dash_bootstrap_components as dbc
import plotly.express as px

from dash import Input
from dash import Output
from dash import State
from dash import callback
from dash import dash_table
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from style import CONTENT_STYLE
from style import NESTA_COLOURS

from dsp_interview_transcripts.utils.data_cleaning import clean_data
from utils.dash_utils import *
from utils.dash_utils import read_data
from utils.topic_modelling import MODEL
from utils.topic_modelling import get_topics_and_summaries


dash.register_page(__name__, path="/topic_modelling", name="Visualisation")

layout = html.Div(
    [  # === Topic modeling controls ===
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Topic Modelling", className="card-title"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Number of topics"),
                                    dcc.Input(
                                        id="num-topics-input",
                                        type="number",
                                        min=2,
                                        max=50,
                                        step=1,
                                        value=10,
                                        style={"width": "100%"},
                                    ),
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label(" "),
                                    dbc.Button(
                                        "Run Topic Model", id="run-topic-model-btn", color="primary", className="mt-2"
                                    ),
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label(" "),
                                    dbc.Spinner(
                                        html.Div(id="topic-model-status"), size="sm", color="info", type="border"
                                    ),
                                ],
                                width=6,
                            ),
                        ]
                    ),
                ]
            ),
            style={"marginBottom": "20px"},
        ),
        # === Display the results of topic modelling ===
        html.Div(
            children=[
                html.P(
                    "This tab contains an interactive visualisation to help you explore user responses within each topic. Each user response is shown as a point."
                ),
                html.P(
                    "Click a point on the plot to find out more information about it. On the left, you will see information about the topic it is in, "
                    "the ID of the conversation it occurred in, and the response itself."
                ),
            ]
        ),
        # First row: Information Panel and Scatterplot
        html.Div(
            [
                # Information panel (left one-third)
                html.Div(
                    id="info-panel",
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "10px",
                        "borderRight": "1px solid #ccc",
                        "backgroundColor": "#f9f9f9",
                    },
                    children=[
                        html.H4("Selected Point Info"),
                        html.Div(id="name-display", style={"marginBottom": "10px"}),
                        html.Div(id="description-display", style={"marginBottom": "10px"}),
                        html.Div(id="conversation-display", style={"marginBottom": "10px"}),
                        html.Div(id="text-clean-display", style={"marginBottom": "10px"}),
                    ],
                ),
                # Scatterplot (right two-thirds)
                html.Div(
                    [dcc.Graph(id="scatter-plot")],
                    style={"width": "75%", "display": "inline-block"},
                ),
            ],
            style={"width": "100%", "marginBottom": "20px"},
        ),
        html.Div(
            children=[
                html.P("When you click a point on the plot, you will see the full text of that conversation below."),
                html.Div(id="conversation-view", style={"marginTop": "20px"}),
            ]
        ),
    ],
    style={**CONTENT_STYLE, "width": "75%", "margin": "0", "padding": "0"},
)


@callback(
    Output("topic-model-status", "children"),
    Output("stored-topic-viz", "data"),  # dataframe for scatterplot
    Input("run-topic-model-btn", "n_clicks"),
    State("num-topics-input", "value"),
    State("stored-data", "data"),  # contents
    State("stored-column-info", "data"),
    prevent_initial_call=True,
)
def run_topic_model(n_clicks, num_topics, contents, column_info):
    """
    Run the topic model, taking as input the uploaded data and the number of topics.
    """
    if not n_clicks or not num_topics:
        raise PreventUpdate

    # Show immediate feedback
    status = f"Running topic model with {num_topics} topics..."

    df_original = read_data(contents)
    conv_id, role_col, text_col, uuid_col = (
        column_info["conv_id"],
        column_info["role_col"],
        column_info["text_col"],
        column_info["uuid_col"],
    )

    interviews_df = clean_data(df_original)

    user_messages = interviews_df[(interviews_df[role_col] == "USER")]

    df_vis, topic_lookup = get_topics_and_summaries(user_messages, text_col, num_topics)

    # After processing
    return dbc.Alert(f"Topic model completed with maximum {num_topics} topics!", color="success"), df_vis.to_dict(
        "records"
    )


@callback(
    Output("scatter-plot", "figure"),
    Input("stored-topic-viz", "data"),
    State("stored-column-info", "data"),
    prevent_initial_call=True,
)
def update_scatter_plot(data, column_info):
    """
    Populate the scatterplot once the topic model has run
    """

    if not data:
        raise PreventUpdate

    conv_id, role_col, text_col, uuid_col = (
        column_info["conv_id"],
        column_info["role_col"],
        column_info["text_col"],
        column_info["uuid_col"],
    )

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=f"{MODEL}_name",
        hover_data=[conv_id, text_col],
        custom_data=[conv_id, text_col, uuid_col],
        color_discrete_sequence=NESTA_COLOURS,
    )

    # Update hovertemplate to show only the text
    fig.update_traces(hovertemplate="<b>%{customdata[1]}</b><extra></extra>")

    fig.update_layout(
        uirevision="scatter-plot",  # Ensure that the zoom level is preserved after you've clicked a point
        xaxis=dict(showticklabels=False, title_text=""),  # Hide x-axis ticks and title
        yaxis=dict(showticklabels=False, title_text=""),  # Hide y-axis ticks and title
        legend_title_text="",  # Hide legend title
    )

    return fig


@dash.callback(
    [
        Output("name-display", "children"),
        Output("description-display", "children"),
        Output("conversation-display", "children"),
        Output("text-clean-display", "children"),
    ],
    Input("scatter-plot", "clickData"),
    Input("stored-topic-viz", "data"),
    State("stored-column-info", "data"),
)
def update_point_info(clickData, data, column_info):
    """
    When the user clicks a point on the scatterplot, show information
    about this point in the left panel.
    """
    if not data:
        raise PreventUpdate

    conv_id, role_col, text_col, uuid_col = (
        column_info["conv_id"],
        column_info["role_col"],
        column_info["text_col"],
        column_info["uuid_col"],
    )

    if clickData:

        df = pd.DataFrame(data)

        selected_uuid = clickData["points"][0]["customdata"][2]
        conversation_id = clickData["points"][0]["customdata"][0]

        selected_point = df[df[uuid_col] == selected_uuid].iloc[0]
        name = f"Topic name: {selected_point[f'{MODEL}_name']}"
        description = f"Topic description: {selected_point.get(f'{MODEL}_description', 'N/A')}"
        conversation = f"Conversation ID: {conversation_id}"
        text_clean = f"User response: {selected_point[text_col]}"

        return name, description, conversation, text_clean

    return "Topic name: N/A", "Topic description: N/A", "Conversation ID: N/A", "User response: N/A"


@callback(
    Output("conversation-view", "children"),
    Input("scatter-plot", "clickData"),
    State("stored-data", "data"),  # contents
    State("stored-column-info", "data"),
)
def display_conversation(clickData, contents, column_info):
    """
    Display the full conversation that the clicked point occurred in,
    with the clicked point text highlighted yellow.
    """
    if not contents or not clickData:
        raise PreventUpdate

    df = read_data(contents)
    conv_id_col = column_info["conv_id"]
    text_col = column_info["text_col"]
    uuid_col = column_info["uuid_col"]
    role_col = column_info["role_col"]

    # Get selected point info
    selected_uuid = clickData["points"][0]["customdata"][2]
    selected_conversation = clickData["points"][0]["customdata"][0]

    # Get all rows in that conversation
    conv_rows = df[df[conv_id_col] == selected_conversation]

    # Format display with highlight on selected uuid
    conversation_display = []
    for i, row in conv_rows.iterrows():
        text = row[text_col]
        role = row[role_col]
        is_selected = row[uuid_col] == selected_uuid
        content = html.Mark(text) if is_selected else text
        conversation_display.append(html.Div([html.Strong(f"{role}: "), html.Span(content)]))

    return conversation_display
