import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table as dt
from dash_table.Format import Format, Scheme, Sign
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h'),
        colorway=["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                  "#0072B2", "#D55E00", "#CC79A7", "#999999"]
    )
)
pio.templates.default = 'custom'

plot_style = {'height': '80vh'}


# Read data
accounts = pd.read_csv('accounts.csv')
bp_accounts = (
    accounts[accounts['CD_CONTA'].str[:1].isin(['1', '2'])]
    .melt('CD_CONTA')
    .dropna()
)['value'].unique()
companies = pd.read_csv('data/companies.csv')
fin_stmts = pd.read_csv('data/fin_stmts_wide.csv')
screener = (
    fin_stmts
    .sort_values(['CD_CVM', 'DT_FIM_EXERC'])
    .groupby('CD_CVM').tail(4)
    .groupby('CD_CVM').sum()
)
for col in bp_accounts:
    screener[col] = screener[col] / 4


def calc_kpis(df):
    df['ShareholderEquity'] = \
        df['StakeholderEquity'] - df['MinorityInterests'].fillna(0)
    df['NetDebt'] = df['Debt'].fillna(0) - df['Cash']
    df['InvestedCapital'] = df['ShareholderEquity'] + df['Debt'].fillna(0)
    
    df['ROE'] = 100 * df['NetIncome'] / df['ShareholderEquity']
    df['ROIC'] = 100 * df['EBIT'] / df['InvestedCapital']
    df['GrossMargin'] = 100 * df['GrossProfit'] / df['Revenue']
    df['EBITMargin'] = 100 * df['EBIT'] / df['Revenue']
    df['NetMargin'] = 100 * df['NetIncome'] / df['Revenue']
    df['NetDebtToEBIT'] = 100 * df['NetDebt'] / df['EBIT']
    return df

screener = calc_kpis(screener)
screener = companies.merge(screener, on='CD_CVM')
screener = screener.sort_values('PESO', ascending=False)

#
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
app.title = 'Análise Fundamentalista'

#
navbar = dbc.NavbarSimple(
    children=[],
    brand=app.title,
    brand_href="#",
    color='dark',
    dark=True
)


# SCREENER MODAL
visible_cols = [
    'TICKER', 'SETOR', 'Revenue', 'EBIT', 'NetIncome',
    'EBITMargin', 'NetMargin', 'ROIC', 'ROE', 'NetDebtToEBIT'
]
screener_table = dt.DataTable(
    id='screener_table',
    data=screener.to_dict('records'),
    columns=[{"name": i, "id": i} for i in screener.columns],
    hidden_columns=[i for i in screener.columns if i not in visible_cols],
    selected_rows=[0],
    style_as_list_view=True,
    style_header={'fontWeight': 'bold'},
    row_selectable='single',
    sort_action='native',
    filter_action='native',
    page_action='native',
    page_size=15
)

#
def grid(rows):
    return html.Div([
        dbc.Row([dbc.Col(col, width=12/len(row)) for col in row])
        for row in rows
    ])


# TABS
tabs = dbc.Tabs([
    dbc.Tab([
        grid([
            [
                dcc.Graph(id='ov_revenue_plot'),
                dcc.Graph(id='ov_profit_plot')
            ],
            [
                dcc.Graph(id='ov_margins_plot'),
                dcc.Graph(id='ov_returns_plot')
            ]
        ])
    ], label="Visão Geral")
])


#
stores = html.Div([dcc.Store(id=f"{s}_store") for s in ['stmts']])


# LAYOUT
app.layout = html.Div([
    dcc.Location('url', refresh=False),
    navbar,
    html.Div(id='page_content', className='container-fluid'),
    stores
])


@app.callback(
    [Output("page_content", "children"),
     Output("stmts_store", "data")],
    [Input('url', 'pathname')]
)
def update_url(url):
    if url is None or len(url) < 4:
        ticker = ''
    else:
        ticker = url[1:5].upper()
        if screener['TICKER'].str[:4].isin([ticker]).sum() == 0:
            ticker = ''
    if ticker == '':
        content = html.Div([
            html.H2('Screener'),
            grid([[screener_table]])
        ])
        data = {}
    else:
        row = screener[screener['TICKER'].str[:4] == ticker]
        cvm_id = row['CD_CVM'].iloc[0]
        company_name = row['NM_PREGAO'].iloc[0]
        sectors = [
            html.Li(row[s].iloc[0], className='breadcrumb-item')
            for s in ['SETOR', 'SUBSETOR', 'SEGMENTO']
        ]
        # Prepare fin statements dataset
        data = fin_stmts[fin_stmts['CD_CVM'] == cvm_id]
        data = data.reset_index()
        data = data[1:]
        data = calc_kpis(data)
        #
        content = html.Div([
            html.H2(company_name),
            html.Ol(sectors, className='breadcrumb',
                style={'background': 'none'}),
            tabs
        ])
        data = data.to_dict('records')
    return content, data


@app.callback(
    [Output('ov_revenue_plot', 'figure'),
     Output('ov_profit_plot', 'figure'),
     Output('ov_margins_plot', 'figure'),
     Output('ov_returns_plot', 'figure')],
    [Input('stmts_store', 'data')]
)
def update_overview_plot(data):
    df = pd.DataFrame(data)
    labs = {'value': '', 'DT_FIM_EXERC': '', 'variable': ''}
    revenue_fig = px.bar(
        df[['DT_FIM_EXERC', 'Revenue', 'GrossProfit']].melt('DT_FIM_EXERC'),
        x='DT_FIM_EXERC', y='value', color='variable', barmode='group',
        title='Receita e Lucro Bruto', labels=labs
    )
    profit_fig = px.bar(
        df[['DT_FIM_EXERC', 'EBIT', 'NetIncome']].melt('DT_FIM_EXERC'),
        x='DT_FIM_EXERC', y='value', color='variable', barmode='group',
        title='Lucro', labels=labs
    )
    margins_fig = px.line(
        df[['DT_FIM_EXERC', 'EBITMargin', 'NetMargin', 'GrossMargin']]
            .melt('DT_FIM_EXERC'),
        x='DT_FIM_EXERC', y='value', color='variable',
        title='Margens', labels=labs
    )
    returns_fig = px.line(
        df[['DT_FIM_EXERC', 'ROIC', 'ROE']].melt('DT_FIM_EXERC'),
        x='DT_FIM_EXERC', y='value', color='variable',
        title='Rentabilidade', labels=labs
    )
    return revenue_fig, profit_fig, margins_fig, returns_fig


#
if __name__ == "__main__":
    app.run_server(debug=True)
