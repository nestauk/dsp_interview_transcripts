import uuid

import dash
import dash_bootstrap_components as dbc

from dash import Input
from dash import Output
from dash import State
from dash import callback
from dash import dcc
from dash import html

from utils.dash_utils import read_data


dash.register_page(__name__, path="/", name="Upload")


layout = html.Div(
    [
        html.H3("Upload Interview CSV"),
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
        html.Div(id="column-selectors"),
    ]
)


@callback(
    Output("upload-feedback", "children"),  # html text to say that the file has successfully been uploaded
    Output("stored-data", "data"),  # store "contents"
    Output("session-id", "data"),  # store session ID
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def handle_upload(contents, filename):
    """Display filename and store uploaded data.

    Create session ID.
    """
    if contents is None:
        return "", None
    # generate new session ID
    session_id = str(uuid.uuid4())
    return f"Uploaded file: {filename}", contents, session_id


@callback(Output("column-selectors", "children"), Input("upload-data", "contents"))
def show_column_selectors(contents):
    """Show column dropdowns after upload and store nothing initially."""
    if not contents:
        return ""

    df = read_data(contents)
    options = [{"label": col, "value": col} for col in df.columns]

    column_ui = html.Div(
        [
            html.H5("Step 1: Select relevant columns"),
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id="conv-id-col", options=options, placeholder="Conversation ID column")),
                    dbc.Col(dcc.Dropdown(id="role-col", options=options, placeholder="Role column")),
                    dbc.Col(dcc.Dropdown(id="text-col", options=options, placeholder="Text column")),
                    dbc.Col(
                        dcc.Dropdown(
                            id="uuid-col",
                            options=[{"label": "None", "value": "None"}] + options,
                            placeholder="Unique ID (optional)",
                        )
                    ),
                ]
            ),
        ]
    )

    return column_ui


@callback(
    Output("stored-column-info", "data"),
    Input("conv-id-col", "value"),
    Input("role-col", "value"),
    Input("text-col", "value"),
    Input("uuid-col", "value"),
)
def store_column_selection(conv_id, role_col, text_col, uuid_col):
    """Store the selected column names into dcc.Store."""
    if not all([conv_id, role_col, text_col]):
        return None

    return {
        "conv_id": conv_id,
        "role_col": role_col,
        "text_col": text_col,
        "uuid_col": uuid_col,
    }
