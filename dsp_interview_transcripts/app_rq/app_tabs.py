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

from dash import ALL
from dash import Input
from dash import Output
from dash import State
from dash import ctx
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from layout.rq_tab import quotes_modal
from layout.rq_tab import rq_tab
from layout.topic_mapping import topic_tab

# tab layouts
from layout.upload import upload_tab
from style import NESTA_COLOURS

from dsp_interview_transcripts.utils.data_cleaning import clean_data
from utils.dash_utils import get_cleaned_data
from utils.dash_utils import get_or_create_output_dir
from utils.llm_question_answering import concat_batch_check_output
from utils.llm_question_answering import normalize_uuid
from utils.llm_question_answering import run_batch_check_for_all_rqs
from utils.llm_summarize import create_output_excel
from utils.llm_summarize import generate_full_summary_output
from utils.llm_summarize import generate_summaries
from utils.topic_modelling import MODEL
from utils.topic_modelling import get_topics_and_summaries


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Multi-Tab Interview Analysis"

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Thematic AI", className="ms-2"),
        ]
    ),
    color="#0F294A",
    dark=True,
    className="mb-4",
)

# App layout with two tabs: Upload & Word Count
app.layout = html.Div(
    [
        # Stores to hold the uploaded DataFrame, selected columns, and session
        dcc.Store(id="data-store"),  # stores raw uploaded data
        dcc.Store(id="column-store"),  # stores {'conv_id', 'role', 'text', 'uuid'}
        dcc.Store(id="session-id"),  # session identifier
        dcc.Store(id="stored-topic-viz"),  # dataframe for scatter plot
        dcc.Store(id="stored-output-paths"),  # for RQ analysis
        dcc.Store(id="stored-rqs"),  # for RQ analysis
        dcc.Store(id="output-dir"),  # for RQ analysis
        dbc.Container(
            [
                navbar,
                # Tabs component
                dcc.Tabs(
                    id="tabs",
                    value="tab-upload",
                    children=[
                        dcc.Tab(
                            label="1. Upload & Select",
                            value="tab-upload",
                        ),
                        dcc.Tab(
                            label="2. Topic mapping",
                            value="tab-topic",
                        ),
                        dcc.Tab(
                            label="3. RQ Analysis",
                            value="tab-rq",
                        ),
                    ],
                ),
                html.Div(
                    [
                        # Upload tab content
                        upload_tab,
                        # topic mapping tab
                        topic_tab,
                        # rq analysis tab
                        rq_tab,
                    ],
                ),
                # Modal for quotes
                quotes_modal,
            ],
            fluid=True,
            style={"maxWidth": "1200px"},
        ),
    ]
)

# Callback to switch visible tab
@app.callback(
    Output("tab-upload", "style"), Output("tab-topic", "style"), Output("tab-rq", "style"), Input("tabs", "value")
)
def switch_tab(tab):
    return (
        {"display": "block"} if tab == "tab-upload" else {"display": "none"},
        {"display": "block"} if tab == "tab-topic" else {"display": "none"},
        {"display": "block"} if tab == "tab-rq" else {"display": "none"},
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


# RQ analysis callback
@app.callback(
    Output("analysis-results", "children"),
    Output("stored-output-paths", "data"),
    Output("stored-rqs", "data"),
    Output("output-dir", "data"),
    Output("download-btn-container", "children"),
    Input("run-analysis", "n_clicks"),
    State("session-id", "data"),
    State("column-store", "data"),
    State("rq-textarea", "value"),
    State("test-mode-toggle", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, session_id, colinfo, rq_text, test_mode):
    if not (n_clicks and session_id and colinfo and rq_text):
        raise PreventUpdate
    df = get_cleaned_data(session_id)
    if colinfo["uuid"] not in df.columns:
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        colinfo["uuid"] = "uuid"
    df[colinfo["uuid"]] = df[colinfo["uuid"]].apply(normalize_uuid)
    outdir = get_or_create_output_dir(session_id, test_mode=test_mode)
    output_paths, rq_dict = run_batch_check_for_all_rqs(
        rq_text=rq_text,
        cleaned_df=df,
        output_dir=outdir,
        conv_col=colinfo["conv_id"],
        role_col=colinfo["role"],
        uuid_col=colinfo["uuid"],
    )
    per_rq, long_dfs = generate_summaries(rq_dict, output_paths, outdir, test_mode, colinfo["text"])
    full_df = generate_full_summary_output(rq_dict, long_dfs, per_rq, colinfo["text"])
    full_df.to_csv(os.path.join(outdir, "full_summary.csv"), index=False)
    create_output_excel(full_df, outdir)
    btn = dbc.Button("Download Results", id="trigger-download", className="nesta-button")
    alert = dbc.Alert(
        "Test mode: mock outputs" if test_mode else "LLM processing complete!",
        color="info" if test_mode else "success",
    )
    return alert, output_paths, rq_dict, outdir, btn


@app.callback(
    Output("download-results", "data"),
    Input("trigger-download", "n_clicks"),
    State("output-dir", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, outdir):
    path = os.path.join(outdir, "full_summary.xlsx")
    if not (n_clicks and os.path.exists(path)):
        raise PreventUpdate
    return dcc.send_file(path)


@app.callback(
    Output("analysis-results", "children", allow_duplicate=True),
    Input("output-dir", "data"),
    State("stored-rqs", "data"),
    State("session-id", "data"),
    State("column-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def display_results(outdir, rq_dict, session_id, colinfo):
    if not (outdir and rq_dict):
        return ""
    full_summary = pd.read_csv(os.path.join(outdir, "full_summary.csv"))
    df_orig = get_cleaned_data(session_id)
    uuid_col = colinfo["uuid"]
    children = []
    for _, q in rq_dict.items():
        temp = full_summary[full_summary["question"] == q]
        if temp.empty:
            children.append(html.Div(f"No output for: {q}", style={"color": "red"}))
            continue
        temp = temp[temp["identifier"].isin(df_orig[uuid_col].astype(str))]
        if temp.empty:
            children.append(html.Div(f"No valid quotes for: {q}", style={"color": "orange"}))
            continue
        quote_elems = []
        for _, row in temp.iterrows():
            quote_elems.append(
                html.Div(row["text"], id={"type": "quote", "index": row["identifier"]}, className="quote-block")
            )
        children.append(html.H4(f"RQ: {q}", style={"marginTop": "1rem"}))
        children.append(html.H6("Summary", style={"color": "grey"}))
        children.append(html.P(temp["answer"].iloc[0]))
        children.append(html.H6("Illustrative quotes", style={"color": "grey"}))
        children.append(html.Div(quote_elems))
    return (html.Div(children),)


@app.callback(
    Output("quote-modal", "is_open"),
    Output("modal-body", "children"),
    Input({"type": "quote", "index": ALL}, "n_clicks"),
    State("session-id", "data"),
    State("column-store", "data"),
)
def display_conversation(n_list, session_id, colinfo):
    if not any(n_list):
        raise PreventUpdate
    triggered = ctx.triggered_id
    uid = triggered["index"]
    df = get_cleaned_data(session_id)
    conv_id = colinfo["conv_id"]
    role = colinfo["role"]
    convo = df[df[colinfo["uuid"]] == uid]
    cid = convo[conv_id].iloc[0]
    subset = df[df[conv_id] == cid]
    body = [
        html.Div(
            [
                html.Strong(f"{r[role]}: "),
                html.Span(html.Mark(r["text_clean"]) if r[colinfo["uuid"]] == uid else r["text_clean"]),
            ]
        )
        for _, r in subset.iterrows()
    ]
    return True, html.Div(body)


if __name__ == "__main__":
    app.run(debug=True)
