import dash

from dash import html


dash.register_page(__name__, path="/")

# Sidebar layout
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


layout = html.Div(
    [
        html.H1("Welcome!", className="display-4"),
        # html.P("This is the home page of the QualFML dashboard."),
    ],
    style=CONTENT_STYLE,
)
