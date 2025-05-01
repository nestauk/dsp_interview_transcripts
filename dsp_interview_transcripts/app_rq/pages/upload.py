import uuid

import dash
import dash_bootstrap_components as dbc

from dash import Input
from dash import Output
from dash import State
from dash import callback
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from style import CONTENT_STYLE
from style import NESTA_COLOURS

from dsp_interview_transcripts.utils.data_cleaning import clean_data
from utils.dash_utils import get_or_create_output_dir
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
        html.Br(),
        html.Div(id="save-data-container"),
        html.Div(id="save-success-msg", style={"marginTop": 10, "color": "green"}),
    ],
    style={
        **CONTENT_STYLE,
        "width": "85%",
        "margin": "0",
        "padding": "2rem",
        "fontFamily": "Century Gothic",
        "color": "#0F294A",
    },
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
    # generate new session ID
    session_id = str(uuid.uuid4())
    if contents is None:
        return "", None, session_id

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


# show the save data button
@callback(
    Output("save-data-container", "children"),
    Input("conv-id-col", "value"),
    Input("role-col", "value"),
    Input("text-col", "value"),
)
def show_save_button(conv_id, role_col, text_col):
    """Display 'Save Data' button only when required columns are selected."""
    if not all([conv_id, role_col, text_col]):
        return ""

    return dbc.Button(
        "Save Data",
        id="save-data-btn",
        n_clicks=0,
        className="mt-2 nesta-button",
    )


@callback(
    Output("save-success-msg", "children"),
    Input("save-data-btn", "n_clicks"),
    State("stored-data", "data"),
    State("stored-column-info", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def save_cleaned_data(n_clicks, contents, column_info, session_id):
    if not n_clicks or not contents or not column_info:
        raise PreventUpdate

    df = read_data(contents)
    conv_id, role_col, text_col = column_info["conv_id"], column_info["role_col"], column_info["text_col"]
    uuid_col = column_info.get("uuid_col", "uuid")

    # Fill in UUIDs if needed
    if uuid_col == "None" or not uuid_col:
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        uuid_col = "uuid"

    df[uuid_col] = df[uuid_col].astype(str)

    print(df.head())
    # Clean the data
    cleaned_df = clean_data(df, text_col)

    # Save to session folder
    output_dir = get_or_create_output_dir(session_id)
    cleaned_df.to_csv(f"{output_dir}/cleaned_data.csv", index=False)

    return "✅ Data cleaned and saved! You can now continue to the next page."
