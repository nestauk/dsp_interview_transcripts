import dash
import pandas as pd
import plotly.express as px

from dash import Input
from dash import Output
from dash import dash_table
from dash import dcc
from dash import html

from dsp_interview_transcripts import PROJECT_DIR


# Load data
data = pd.read_csv(PROJECT_DIR / "data/cleaned_data.csv")
transcripts = pd.read_csv(PROJECT_DIR / "data/qual_af_transcripts.csv")
transcripts = (
    transcripts.assign(text=lambda x: x["text"].fillna(x["transcript"]))
    # Length of text
    .fillna({"text": ""}).assign(text_length=lambda x: x["text"].apply(len))
)

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
                                        for topic in data["Name"]
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
        topic_data = data[data["Name"] == selected_topic]
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
        data,
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
