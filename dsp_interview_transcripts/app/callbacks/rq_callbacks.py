import os
import uuid

import dash_bootstrap_components as dbc
import pandas as pd

from dash import ALL
from dash import Input
from dash import Output
from dash import State
from dash import ctx
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate

from utils.dash_utils import get_cleaned_data
from utils.dash_utils import get_or_create_output_dir
from utils.llm_question_answering import concat_batch_check_output
from utils.llm_question_answering import normalize_uuid
from utils.llm_question_answering import run_batch_check_for_all_rqs
from utils.llm_summarize import create_output_excel
from utils.llm_summarize import generate_full_summary_output
from utils.llm_summarize import generate_summaries


def register_rq_callbacks(app):
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
