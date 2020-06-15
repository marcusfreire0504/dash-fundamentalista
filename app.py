import pandas as pd

import dash
import dash_bootstrap_components as dbc

import plotly.io as pio
import plotly.graph_objects as go

colorscheme = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
               "#0072B2", "#D55E00", "#CC79A7", "#999999"]
pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h'),
        colorway=colorscheme
    )
)
pio.templates.default = 'custom'

plot_style = {'height': '80vh'}


# Read data
fin_stmts = pd.read_csv('data/fin_stmts_wide.csv')
companies = pd.read_csv('data/companies.csv')
companies = companies[companies['CD_CVM'].isin(fin_stmts['CD_CVM'])]


#
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
app.title = 'Análise Fundamentalista'

