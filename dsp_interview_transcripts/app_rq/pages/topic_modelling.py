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
from umap import UMAP

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts.pipeline.process_data import create_context
from dsp_interview_transcripts.pipeline.process_data import get_sentiment
from dsp_interview_transcripts.utils.data_cleaning import clean_data
from dsp_interview_transcripts.utils.llm_utils import *
from dsp_interview_transcripts.utils.topic_modelling import embed_docs
from dsp_interview_transcripts.utils.topic_modelling import init_topic_model
from utils.dash_utils import *
from utils.dash_utils import read_data


dash.register_page(__name__, path="/topic_modelling", name="Visualisation")


class NameDescription(BaseModel):
    """Model for naming and describing a group of documents."""

    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


MODEL = "llama3.2"

llm_chain = get_chain(
    PROJECT_DIR / "dsp_interview_transcripts/pipeline/prompts/basic_prompt.txt",
    input_vars=["docs", "keywords"],
    output_template=NameDescription,
    provider="ollama",
    model=MODEL,
    temp=0,
)

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

# Run the topic model and track status
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

    docs = user_messages[text_col].tolist()
    docs, embeddings = embed_docs(docs, save=False)

    topic_model, vectorizer_model, representation_model = init_topic_model(
        stop_words="english",
        min_cluster_size=10,
        hdbscan_selection_method="leaf",
        embedding_model="all-MiniLM-L6-v2",
        seed=42,
        empty_reduction=False,
        nr_topics=10,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)

    new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")

    topic_model.update_topics(
        docs,
        topics=new_topics,
        top_n_words=10,
        n_gram_range=(1, 3),
        vectorizer_model=vectorizer_model,
        ctfidf_model=None,
        representation_model=representation_model,
    )

    summary_info = topic_model.get_topic_info()

    results = name_topics(
        summary_info,
        llm_chain,
        input_variable_dict={"docs": "Representative_Docs", "keywords": "Representation"},
        topic_label_col="Topic",
    )

    topic_info = format_output_df(
        topic_info=summary_info, results=results, output_fields=["name", "description"], model_name=MODEL
    )

    umap_2d = UMAP(random_state=42, n_components=2)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    topic_lookup = topic_info[["Topic", "Name", "Representation", f"{MODEL}_name", f"{MODEL}_description"]]

    df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_vis["topic"] = new_topics
    df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
    df_vis["doc"] = docs

    df_vis = pd.merge(
        user_messages,
        df_vis,
        left_on=text_col,
        right_on="doc",
        how="outer",
    )

    # After processing
    return dbc.Alert(f"Topic model completed with {num_topics} topics!", color="success"), df_vis.to_dict("records")


@callback(
    Output("scatter-plot", "figure"),
    Input("stored-topic-viz", "data"),
    State("stored-column-info", "data"),
    prevent_initial_call=True,
)
def update_scatter_plot(data, column_info):

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


# Callback to update "Selected Point Info" panel
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


# Display the conversation that the clicked point occurred in
@callback(
    Output("conversation-view", "children"),
    Input("scatter-plot", "clickData"),
    State("stored-data", "data"),  # contents
    State("stored-column-info", "data"),
)
def display_conversation(clickData, contents, column_info):
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
