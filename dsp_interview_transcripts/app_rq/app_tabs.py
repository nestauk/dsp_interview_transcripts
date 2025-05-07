import base64
import io
import os
import re
import uuid

from collections import Counter

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from dash import Input
from dash import Output
from dash import State
from dash import dash_table
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from style import NESTA_COLOURS

from dsp_interview_transcripts.utils.data_cleaning import clean_data
from utils.dash_utils import get_cleaned_data
from utils.dash_utils import get_or_create_output_dir
from utils.topic_modelling import MODEL
from utils.topic_modelling import get_topics_and_summaries


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Multi-Tab Text Analysis"

# App layout with two tabs: Upload & Word Count
app.layout = html.Div(
    [
        # Stores to hold the uploaded DataFrame, selected columns, and session
        dcc.Store(id="data-store"),  # stores raw uploaded data
        dcc.Store(id="column-store"),  # stores {'conv_id', 'role', 'text', 'uuid'}
        dcc.Store(id="session-id"),  # session identifier
        dcc.Store(id="stored-topic-viz"),  # dataframe for scatter plot
        # Tabs component
        dcc.Tabs(
            id="tabs",
            value="tab-upload",
            children=[
                dcc.Tab(label="1. Upload & Select", value="tab-upload"),
                dcc.Tab(label="2. Topic mapping", value="tab-topic"),
            ],
        ),
        # Upload tab content
        html.Div(
            id="tab-upload",
            children=[
                html.H3("Upload CSV"),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select a CSV File")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                    multiple=False,
                ),
                html.Div(id="upload-feedback", style={"marginTop": 10}),
                html.Hr(),
                # Column selectors
                html.Div(
                    [
                        html.H5("Select columns"),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Dropdown(id="conv-id-dropdown", placeholder="Conversation ID column")),
                                dbc.Col(dcc.Dropdown(id="role-dropdown", placeholder="Role column")),
                                dbc.Col(dcc.Dropdown(id="text-dropdown", placeholder="Text column")),
                                dbc.Col(dcc.Dropdown(id="uuid-dropdown", placeholder="UUID column")),
                            ]
                        ),
                        html.Br(),
                        dbc.Button("Save", id="save-btn", color="primary"),
                        html.Div(id="save-feedback", style={"marginTop": 10, "color": "green"}),
                    ],
                    id="column-section",
                    style={"display": "none"},
                ),
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="tab-topic",
            children=[
                html.H3("Topic Mapping"),
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Number of topics"),
                                        dcc.Input(
                                            id="num-topics-input",
                                            type="number",
                                            min=2,
                                            max=100,
                                            step=1,
                                            value=10,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Button("Run", id="run-topic-model-btn", className="mt-2 nesta-button"), width=2
                                ),
                                dbc.Col(
                                    dbc.Spinner(html.Div(id="topic-model-status"), size="sm", color="info"), width=7
                                ),
                            ]
                        )
                    ),
                    className="mb-3",
                ),
                html.Div(
                    id="topic-results",
                    style={
                        "display": "none"
                    },  # by default, the results are hidden. They are revealed once the model has run.
                    children=[
                        html.Div(
                            [
                                html.H4(
                                    "Topic descriptions and key words",
                                    style={"color": "#0F294A", "fontWeight": "bold", "marginTop": "20px"},
                                ),
                                dash_table.DataTable(
                                    id="topic-lookup-table",
                                    page_action="none",
                                    style_table={"height": "500px", "overflowY": "auto", "overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "Century Gothic",
                                        "fontSize": "14px",
                                        "textAlign": "left",
                                    },
                                    style_header={
                                        "backgroundColor": "#0F294A",
                                        "color": "white",
                                        "fontWeight": "bold",
                                        "textAlign": "center",
                                    },
                                    style_data={
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                        "fontFamily": "Century Gothic",
                                        "fontSize": "14px",
                                        "color": "#0F294A",
                                    },
                                    sort_action="native",
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Download Topics as CSV",
                                    id="download-topic-csv-btn",
                                    className="nesta-button",
                                ),
                                dcc.Download(id="download-topic-csv"),
                            ]
                        ),
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
                                        "padding": "20px",
                                        "borderRight": "2px solid #ccc",
                                        "backgroundColor": "#F6F8FA",
                                        "fontFamily": "Century Gothic",
                                        "fontSize": "14px",
                                        "color": "#0F294A",
                                    },
                                    children=[
                                        html.H4(
                                            "Selected Point Info", style={"color": "#0F294A", "fontWeight": "bold"}
                                        ),
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
                                html.P(
                                    "When you click a point on the plot, you will see the full text of that conversation below."
                                ),
                                html.Div(id="conversation-view", style={"marginTop": "20px"}),
                            ]
                        ),
                    ],
                ),
            ],
            style={"display": "none"},
        ),
    ]
)

# Callback to switch visible tab
@app.callback(Output("tab-upload", "style"), Output("tab-topic", "style"), Input("tabs", "value"))
def switch_tab(tab):
    return (
        {"display": "block"} if tab == "tab-upload" else {"display": "none"},
        {"display": "block"} if tab == "tab-topic" else {"display": "none"},
    )


# Callback to parse upload and reset state
@app.callback(
    Output("upload-feedback", "children"),
    Output("data-store", "data"),
    Output("session-id", "data"),
    Output("column-store", "data"),
    Output("save-feedback", "children"),
    Output("conv-id-dropdown", "value"),
    Output("role-dropdown", "value"),
    Output("text-dropdown", "value"),
    Output("uuid-dropdown", "value"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, stored_sid):
    if not contents:
        raise PreventUpdate
    # Decode and read CSV
    content_string = contents.split(",")[1]
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    # Preserve or generate session ID
    session_id = stored_sid or str(uuid.uuid4())
    # Reset column selections and feedback
    feedback = f"Uploaded: {filename}"
    return (
        feedback,
        df.to_json(date_format="iso", orient="split"),
        session_id,
        None,  # clear column-store
        "",  # clear save-feedback
        None,
        None,
        None,
        None,  # reset dropdown values
    )


# Callback to show column dropdowns once data is uploaded
@app.callback(
    Output("column-section", "style"),
    Output("conv-id-dropdown", "options"),
    Output("role-dropdown", "options"),
    Output("text-dropdown", "options"),
    Output("uuid-dropdown", "options"),
    Input("data-store", "data"),
)
def show_column_selector(jsonified_data):
    if not jsonified_data:
        return {"display": "none"}, [], [], [], []
    df = pd.read_json(jsonified_data, orient="split")
    opts = [{"label": col, "value": col} for col in df.columns]
    return {"display": "block"}, opts, opts, opts, opts


# Callback to save selected columns, clean and save data
@app.callback(
    Output("save-feedback", "children", allow_duplicate=True),
    Output("column-store", "data", allow_duplicate=True),
    Input("save-btn", "n_clicks"),
    State("data-store", "data"),
    State("conv-id-dropdown", "value"),
    State("role-dropdown", "value"),
    State("text-dropdown", "value"),
    State("uuid-dropdown", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_columns_and_clean(n, json_data, conv_col, role_col, text_col, uuid_col, session_id):
    if not (n and json_data and conv_col and role_col and text_col and uuid_col and session_id):
        raise PreventUpdate
    df = pd.read_json(json_data, orient="split")
    # Ensure UUID column
    if uuid_col not in df.columns or df[uuid_col].isnull().any():
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        uuid_col = "uuid"
    df[uuid_col] = df[uuid_col].astype(str)
    # Clean data
    cleaned_df = clean_data(df, text_col)
    # Save cleaned data
    output_dir = get_or_create_output_dir(session_id)
    cleaned_df.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
    # Store columns mapping
    cols = {"conv_id": conv_col, "role": role_col, "text": text_col, "uuid": uuid_col}
    return ("✅ Columns saved and data cleaned!", cols)


# ----- Topic mapping --------------------------------------------

# Topic modelling callback
@app.callback(
    Output("topic-model-status", "children"),
    Output("stored-topic-viz", "data"),
    Output("topic-lookup-table", "data"),
    Output("topic-lookup-table", "columns"),
    Output("topic-results", "style"),
    Input("run-topic-model-btn", "n_clicks"),
    State("num-topics-input", "value"),
    State("column-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def run_topic_model(n_clicks, num_topics, colinfo, session_id):
    if not n_clicks or not num_topics or not colinfo or not session_id:
        raise PreventUpdate
    status_msg = f"Running topic model with {num_topics} topics..."
    # Load cleaned data
    output_dir = get_or_create_output_dir(session_id)
    df = get_cleaned_data(session_id)
    # Filter user messages
    user_msgs = df[df[colinfo["role"]] == "USER"]
    # Run topic model
    df_vis, topic_lookup = get_topics_and_summaries(user_msgs, colinfo["text"], num_topics)
    # Save lookup
    topic_lookup.to_csv(os.path.join(output_dir, "topic_lookup.csv"), index=False)
    # Prepare table data
    topic_lookup = topic_lookup.rename(
        columns={
            "llama3.2_name": "Topic name",
            "llama3.2_description": "Description",
            "Representation": "Keywords",
        }
    )[["Topic", "Topic name", "Description", "Keywords"]]

    # **Convert each list of keywords into a single string**:
    topic_lookup["Keywords"] = topic_lookup["Keywords"].apply(
        lambda kws: ", ".join(kws) if isinstance(kws, (list, tuple)) else str(kws)
    )

    table_data = topic_lookup.to_dict("records")
    table_cols = [{"name": c, "id": c} for c in topic_lookup.columns]
    # Show results
    return (
        dbc.Alert(f"Topic model completed with up to {num_topics} topics", color="success"),
        df_vis.to_dict("records"),
        table_data,
        table_cols,
        {"display": "block"},
    )


# Download CSV callback
@app.callback(
    Output("download-topic-csv", "data"),
    Input("download-topic-csv-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_topic_csv(n_clicks, session_id):
    if not n_clicks or not session_id:
        raise PreventUpdate
    path = os.path.join(get_or_create_output_dir(session_id), "topic_lookup.csv")
    if not os.path.exists(path):
        raise PreventUpdate
    return dcc.send_file(path)


@app.callback(
    Output("scatter-plot", "figure"),
    Input("stored-topic-viz", "data"),
    State("column-store", "data"),
    prevent_initial_call=True,
)
def update_scatter_plot(data, column_info):
    """
    Populate the scatterplot once the topic model has run
    """

    if not data:
        raise PreventUpdate

    print(column_info)

    conv_id, role_col, text_col, uuid_col = (
        column_info["conv_id"],
        column_info["role"],
        column_info["text"],
        column_info["uuid"],
    )

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=f"{MODEL}_name",
        hover_data=[conv_id, "text_clean"],
        custom_data=[conv_id, "text_clean", uuid_col],
        color_discrete_sequence=NESTA_COLOURS,
    )

    # Update hovertemplate to show only the text
    fig.update_traces(hovertemplate="<b>%{customdata[1]}</b><extra></extra>")

    fig.update_layout(
        uirevision="scatter-plot",  # Ensure that the zoom level is preserved after you've clicked a point
        xaxis=dict(showticklabels=False, title_text=""),  # Hide x-axis ticks and title
        yaxis=dict(showticklabels=False, title_text=""),  # Hide y-axis ticks and title
        legend_title_text="",  # Hide legend title
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1),
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(family="Century Gothic", size=12),
    )

    return fig


@app.callback(
    [
        Output("name-display", "children"),
        Output("description-display", "children"),
        Output("conversation-display", "children"),
        Output("text-clean-display", "children"),
    ],
    Input("scatter-plot", "clickData"),
    Input("stored-topic-viz", "data"),
    State("column-store", "data"),
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
        column_info["role"],
        column_info["text"],
        column_info["uuid"],
    )

    if clickData:

        df = pd.DataFrame(data)

        selected_uuid = clickData["points"][0]["customdata"][2]
        conversation_id = clickData["points"][0]["customdata"][0]

        selected_point = df[df[uuid_col] == selected_uuid].iloc[0]
        name = f"Topic name: {selected_point[f'{MODEL}_name']}"
        description = f"Topic description: {selected_point.get(f'{MODEL}_description', 'N/A')}"
        conversation = f"Conversation ID: {conversation_id}"
        text_clean = f"User response: {selected_point['text_clean']}"

        return name, description, conversation, text_clean

    return "Topic name: N/A", "Topic description: N/A", "Conversation ID: N/A", "User response: N/A"


@app.callback(
    Output("conversation-view", "children"),
    Input("scatter-plot", "clickData"),
    State("session-id", "data"),
    State("column-store", "data"),
)
def display_conversation(clickData, session_id, column_info):
    """
    Display the full conversation that the clicked point occurred in,
    with the clicked point text highlighted yellow.
    """
    if not session_id or not clickData:
        raise PreventUpdate

    df = get_cleaned_data(session_id)

    conv_id_col = column_info["conv_id"]
    text_col = column_info["text"]
    uuid_col = column_info["uuid"]
    role_col = column_info["role"]

    # Get selected point info
    selected_uuid = clickData["points"][0]["customdata"][2]
    selected_conversation = clickData["points"][0]["customdata"][0]

    # Get all rows in that conversation
    conv_rows = df[df[conv_id_col] == selected_conversation]

    # Format display with highlight on selected uuid
    conversation_display = []
    for i, row in conv_rows.iterrows():
        text = row["text_clean"]
        role = row[role_col]
        is_selected = row[uuid_col] == selected_uuid
        content = html.Mark(text) if is_selected else text
        conversation_display.append(html.Div([html.Strong(f"{role}: "), html.Span(content)]))

    return conversation_display


if __name__ == "__main__":
    app.run(debug=True)
