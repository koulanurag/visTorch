# -*- coding: utf-8 -*-
import random
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from visTorch.components import autoencoder


class VisBoard:
    def __init__(self):
        """Creates and initializes a Plotly Dash App to which more visual components could be added"""

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
        """ Adds an AutoEncoder Interface to the app.

        :param model: Pytorch model with functions : encoder(), decoder()
        :param dataset: Dataset for sampling input
        :param latent_options: Dictionary having :
        :param model_paths: Dict of paths for different saved model,
                            Key is the title of the path and Value is the actual path
        :param pre_process: callback function that has to be called for transforming
                            input tensor sampled from the Dataset into an Image

        """
        """ Adds a AutoEncoder interface"""
        prefix = str(random.randint(0, 100000000000))
        ae = autoencoder(self.app, model, dataset, latent_options, model_paths, pre_process, prefix=prefix)
        self.body_children.append(ae)

    def run_server(self, host, port, debug=True):
        """

        :param host: Address to host the app
        :param port: Port for app hosting
        :param debug:
        :return:
        """
        body = dbc.Container(self.body_children)
        self.app.layout = html.Div([self.navbar, body])
        self.app.run_server(host=host, port=port, debug=debug)