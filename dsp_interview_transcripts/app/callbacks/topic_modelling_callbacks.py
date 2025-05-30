import os

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from dash import Input
from dash import Output
from dash import State
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from style import NESTA_COLOURS

from utils.dash_utils import get_cleaned_data
from utils.dash_utils import get_or_create_output_dir
from utils.topic_modelling import MODEL
from utils.topic_modelling import get_topics_and_summaries


# # TODO: better way of identifying user messages
# INTERVIEWER_TERMS = ["BOT", "interviewer"]


def register_topic_callbacks(app):
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
        user_msgs = df
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

        if conv_id is None:
            conv_id = "source_file"

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
