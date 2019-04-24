# -*- coding: utf-8 -*-
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from visTorch.components import autoencoder


class VisBoard:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.config.suppress_callback_exceptions = True
        self.navbar = self.get_navbar(self.app)
        self.body_children = []

    @staticmethod
    def get_navbar(app):
        navbar = dbc.NavbarSimple(
            brand="VisTorch",
            brand_href="#"
        )
        return navbar

    def add_ae(self, model, dataset, latent_options, model_paths, pre_process=None):
        """ Adds a AutoEncoder interface"""
        ae = autoencoder(self.app, model, dataset, latent_options, model_paths, pre_process)
        self.body_children.append(ae)

    def run_server(self, host, port, debug=True):
        body = dbc.Container(self.body_children)
        self.app.layout = html.Div([self.navbar, body])
        self.app.run_server(host=host, port=port, debug=debug)
