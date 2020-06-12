import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from app import app, companies, fin_stmts
from funcs import grid, calc_kpis


def layout(ticker):
    row = companies[companies['BTICKER'].str[:4] == ticker]
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
    return html.Div([
        dcc.Store(id='stmts_store', data=data.to_dict('records')),
        html.H2(company_name),
        html.Ol(sectors, className='breadcrumb',
            style={'background': 'none'}),
        dbc.Tabs([
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
    ])


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

