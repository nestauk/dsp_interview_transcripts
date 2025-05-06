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
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate

from dsp_interview_transcripts.utils.data_cleaning import clean_data
from utils.dash_utils import get_or_create_output_dir


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
        dcc.Store(id="data-store"),
        dcc.Store(id="column-store"),  # stores {'conv_id', 'role', 'text', 'uuid'}
        dcc.Store(id="session-id"),  # session identifier
        # Tabs component
        dcc.Tabs(
            id="tabs",
            value="tab-upload",
            children=[
                dcc.Tab(label="1. Upload & Select", value="tab-upload"),
                dcc.Tab(label="2. Word Count", value="tab-bar"),
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
        # Word count tab content
        html.Div(
            id="tab-bar",
            children=[
                html.H3("Top Words"),
                dcc.Graph(id="bar-plot"),
            ],
            style={"display": "none"},
        ),
    ]
)

# Callback to switch visible tab
@app.callback(Output("tab-upload", "style"), Output("tab-bar", "style"), Input("tabs", "value"))
def switch_tab(tab):
    return (
        {"display": "block"} if tab == "tab-upload" else {"display": "none"},
        {"display": "block"} if tab == "tab-bar" else {"display": "none"},
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


# Callback to generate bar plot from cleaned data
@app.callback(
    Output("bar-plot", "figure"),
    Input("column-store", "data"),
    State("session-id", "data"),
)
def update_bar_plot(colinfo, session_id):
    if not (colinfo and session_id):
        raise PreventUpdate
    path = os.path.join(get_or_create_output_dir(session_id), "cleaned_data.csv")
    if not os.path.exists(path):
        raise PreventUpdate
    df = pd.read_csv(path)
    text_col = colinfo["text"]
    text = df[text_col].dropna().astype(str).str.lower().str.cat(sep=" ")
    words = re.findall(r"\b\w+\b", text)
    count = Counter(words)
    top20 = pd.DataFrame(count.items(), columns=["word", "count"]).nlargest(20, "count")
    fig = px.bar(top20, x="word", y="count", title="Top 20 Words (Cleaned)")
    return fig


if __name__ == "__main__":
    app.run(debug=True)
