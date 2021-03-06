import pandas as pd

import dash
import dash_bootstrap_components as dbc

import plotly.io as pio
import plotly.graph_objects as go

from data_funcs import get_mktcap

colorscheme = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
               "#0072B2", "#D55E00", "#CC79A7", "#999999"]
pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        # Small margins
        margin=dict(l=50, r=20, t=60, b=40),
        # Horizontal legend on top-left
        legend=dict(orientation='h', x=0, y=1.05),
        # Title position on top-left
        title={'y': 0.97, 'x': 0, 'xanchor': 'left', 'yanchor': 'top'},
        # Bigger title
        titlefont={'size': 20},
        # Custom color scheme 
        colorway=colorscheme,
        #
        hovermode='x'
    )
)
pio.templates.default = 'custom'

plot_style = {'height': '80vh'}


# Read data
fin_stmts = pd.read_csv('data/fin_stmts_wide.csv')
companies = pd.read_csv('data/companies.csv')
companies = companies[companies['CD_CVM'].isin(fin_stmts['CD_CVM'])]
companies = companies.merge(get_mktcap(), on="NM_PREGAO", how="inner")

#
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{
        'name': 'viewport',
        'content': "width=device-width, initial-scale=1, shrink-to-fit=no"
    }]
)
server = app.server
app.title = 'Análise Fundamentalista'

