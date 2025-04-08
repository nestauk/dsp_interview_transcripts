import dash

from dash import html

from utils.dash_utils import *


dash.register_page(__name__, path="/topic_modelling", name="Visualisation")

layout = html.Div([html.H3("Visualisation Placeholder"), html.P("This page will contain interactive plots.")])
