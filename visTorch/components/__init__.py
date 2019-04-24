import base64
import random
import torch
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from io import BytesIO


def _img_resize(img):
    img = img.resize((200, 200))
    return img


def autoencoder(app, model, dataset, latent_options, paths, pre_process=None, prefix=""):
    prefix += '-ae-'

    header = [html.Div("Auto Encoder", className="float-left col-md-2"),
              html.Div([dcc.Dropdown(
                             id=prefix + 'selected-model',
                             options=[{"label": k, "value": v} for k, v in paths.items()],
                             value=paths[next(iter(paths.keys()))],
                             searchable=False,
                             className="d-inline w-50 float-right"
                         ),
                        html.Span("Model: ", className="d-inline w-50 float-right")],
                  className="float-right col-md-4")]

    input_div = dbc.Col(
        dbc.Card([dbc.CardHeader([
            "Input",
            dbc.Button('sample', color="info", id=prefix + 'sample-input',
                       className="mr-1 float-right"),
        ]),
            dbc.CardBody(
                [
                    html.Div(children=[], id=prefix + 'input-content'),
                    html.Span(children="", id=prefix + 'input-content-id', className='d-none'),
                ],
                className="d-flex justify-content-center"
            ),
        ]
        )
    )

    latent_size = latent_options['n']
    latent_space = []

    for _ in range(latent_size):
        id = prefix + 'latent-slider-' + str(_)
        latent_space.append(dcc.Slider(min=latent_options['min'],
                                       max=latent_options['max'],
                                       marks={latent_options['min']: latent_options['min'],
                                              latent_options['max']: latent_options['max']},
                                       step=latent_options['step'],
                                       updatemode='drag',
                                       id=id,
                                       value=latent_options['min'],
                                       className="mt-3 mb-3"))
    latent_div = dbc.Col(dbc.Card([dbc.CardHeader("Latent Space"),
                                   html.Span(id=prefix + "hidden-latent-space", children="", className='d-none'),
                                   dbc.CardBody([html.Div(children=latent_space, id=prefix + 'output-latent')]), ]))

    output_div = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader("Output"),
                dbc.CardBody(
                    [
                        html.Div(children=[], id=prefix + 'output-content'),
                    ],
                    className="d-flex justify-content-center"
                ),
            ]

        )
    )

    @app.callback(
        [Output(component_id=prefix + 'input-content-id', component_property='children'),
         Output(component_id=prefix + 'input-content', component_property='children'),
         Output(component_id=prefix + "hidden-latent-space", component_property='children')],
        [Input(component_id=prefix + 'sample-input', component_property='n_clicks')])
    def sample_input(n_clicks):
        input_id = random.randint(0, len(dataset) - 1)
        img, _ = dataset[input_id]

        hx = model.encoder(img.view(-1))
        hx = [round(_, len(str(latent_options['step']).split('.')[-1])) for _ in hx.cpu().data.numpy().tolist()]

        # pre-process to get PIL image
        if pre_process is not None:
            disp_img = pre_process(img)
        else:
            disp_img = img
        disp_img = _img_resize(disp_img)

        # convert to byte code image
        output = BytesIO()
        disp_img.save(output, format='PNG')
        output.seek(0)
        encoded_image = base64.b64encode(output.read())

        html_img = html.Img(src='data:image/png;base64,{}'.format(str(encoded_image)[2:-1]))
        hx = ','.join([str(_) for _ in hx])
        return str(input_id), html_img, hx

    for slider_id in range(latent_size):
        @app.callback(
            Output(component_id=prefix + 'latent-slider-' + str(slider_id), component_property="value"),
            [Input(component_id=prefix + "hidden-latent-space", component_property='children')],
            [State(component_id=prefix + 'latent-slider-' + str(slider_id), component_property="id")])
        def set_latent_slider(latent_space, slider_id):
            slider_id = int(slider_id.split("-")[-1])
            return float(latent_space.split(",")[slider_id])

    @app.callback(
        Output(component_id=prefix + 'output-content', component_property='children'),
        [Input(component_id=prefix + 'latent-slider-' + str(slider_id), component_property='value')
         for slider_id in range(latent_size)])
    def predicted_output(*latent_space):
        hx = torch.FloatTensor(latent_space)
        output = model.decoder(hx)

        # pre-process to get numpy image
        if pre_process is not None:
            img = pre_process(output.reshape(dataset[0][0].shape))
        img = _img_resize(img)
        # convert to img
        output = BytesIO()
        img.save(output, format='PNG')
        output.seek(0)
        encoded_image = base64.b64encode(output.read())

        html_img = html.Img(src='data:image/png;base64,{}'.format(str(encoded_image)[2:-1]))
        return html_img

    ae_div = dbc.Card(
        [dbc.CardHeader(header),
         dbc.CardBody(dbc.Row([input_div, latent_div, output_div], className='m-2'))],
        className="mt-4 mb-4 border-secondary"
    )
    return ae_div
