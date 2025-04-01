import os

import dash
import dash_auth
import dash_bootstrap_components as dbc

from dash import dcc
from dash import html
from dash_bootstrap_templates import load_figure_template
from dotenv import load_dotenv


load_dotenv()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

auth = dash_auth.BasicAuth(app, {os.environ.get("VALID_USERNAME"): os.environ.get("VALID_PASSWORD")})

load_figure_template("BOOTSTRAP")

# Importing all the pages - this has to go AFTER app is initialised
from pages import home
from pages import overview
from pages import scatterplot
from style import CONTENT_STYLE
from style import NESTA_COLOURS
from style import SIDEBAR_STYLE


sidebar = html.Div(
    [
        html.H2("QualFML", className="display-5"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Topic overview", href="/overview", active="exact"),
                # dbc.NavLink("Information by topic", href="/topic_info", active="exact"),
                dbc.NavLink("User response mapping", href="/scatterplot", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Main layout with sidebar and page content
app.layout = html.Div([dcc.Location(id="url"), sidebar, dash.page_container])

if __name__ == "__main__":
    app.run_server(
        # debug=True, # comment out when deploying to production
        host="0.0.0.0",
        port=8050,  # comment this part out when testing on your local machine & on public wifi
    )
