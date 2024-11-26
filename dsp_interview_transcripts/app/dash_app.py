import dash
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

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Tabs(
            [
                # Placeholder for summary topic info
                dcc.Tab(
                    label="Topic Information",
                    children=[
                        html.Div(
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
                    ],
                ),
                # Tab 1: Main Scatter Plot and Table View
                dcc.Tab(
                    label="Scatter Plot & Table",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(id="scatter-plot"),
                                    ],
                                    style={"width": "48%", "display": "inline-block"},
                                ),
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
                                    style={"width": "48%", "display": "inline-block", "verticalAlign": "top"},
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        )
    ]
)


@app.callback(Output("topic-info", "children"), Input("topic-dropdown", "value"))
def display_topic_info(selected_topic):
    if selected_topic:
        topic_data = data_viz[data_viz["Name"] == selected_topic]
        topic_summary = f"Topic: {selected_topic}\nNumber of mentions: {len(topic_data)}"

        return html.Div(
            [
                html.H4(f"Information about {selected_topic}"),
                html.P(topic_summary),
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
