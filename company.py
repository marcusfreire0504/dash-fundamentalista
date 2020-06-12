import numpy as np
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

import statsmodels.api as sm
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
                        dcc.Graph(id='ov_profit_plot'),
                        dcc.Graph(id='ov_debt_plot')
                    ],
                    [
                        dcc.Graph(id='ov_margins_plot'),
                        dcc.Graph(id='ov_returns_plot'),
                        dcc.Graph(id='ov_debt2_plot')
                    ]
                ])
            ], label="Visão Geral"),
            dbc.Tab([
                dbc.RadioItems(
                    id='rev_forecast_method',
                    value='ets',
                    inline=True,
                    options=[
                        {'value': 'ets', 'label': 'Alisamento exponencial'},
                        {'value': 'arima', 'label': 'ARIMA'}
                    ]
                ),
                dcc.Graph("rev_forecast_plot", style={'height': '80vh'})
            ], label="Receita")
        ])
    ])


@app.callback(
    [Output('ov_revenue_plot', 'figure'),
     Output('ov_profit_plot', 'figure'),
     Output('ov_margins_plot', 'figure'),
     Output('ov_returns_plot', 'figure'),
     Output('ov_debt_plot', 'figure'),
     Output('ov_debt2_plot', 'figure')],
    [Input('stmts_store', 'data')]
)
def update_overview_plot(data):
    df = pd.DataFrame(data)
    labs = {'value': '', 'DT_FIM_EXERC': '', 'variable': ''}
    revenue_fig = px.bar(
        df,  barmode='group',
        x='DT_FIM_EXERC', y=['Revenue', 'GrossProfit'], color='variable',
        title='Receita e Lucro Bruto', labels=labs
    )
    profit_fig = px.bar(
        df,  barmode='group',
        x='DT_FIM_EXERC', y=['EBIT', 'NetIncome'], color='variable',
        title='Lucro', labels=labs
    )
    margins_fig = px.line(
        df,
        x='DT_FIM_EXERC', y=['EBITMargin', 'NetMargin', 'GrossMargin'],
        color='variable',
        title='Margens', labels=labs
    )
    returns_fig = px.line(
        df,
        x='DT_FIM_EXERC', y=['ROIC', 'ROE'], color='variable',
        title='Rentabilidade', labels=labs
    )
    debt_fig = px.bar(
        df,
        x='DT_FIM_EXERC', y=['Debt', 'ShareholderEquity'], color='variable',
        title='Estrutura de Capital', labels=labs
    )
    debt2_fig = px.line(
        df,
        x='DT_FIM_EXERC', y=['NetDebtToEBIT', 'DebtToEquity'], color='variable',
        title='Endividamento', labels=labs
    )
    
    return revenue_fig, profit_fig, margins_fig, \
        returns_fig, debt_fig, debt2_fig


@app.callback(
    Output('rev_forecast_plot', 'figure'),
    [Input('stmts_store', 'data'),
     Input('rev_forecast_method', 'value')]
)
def update_revenue_forecast(data, method):
    data = pd.DataFrame(data)
    data['DT_FIM_EXERC'] = pd.to_datetime(data['DT_FIM_EXERC'])
    data = data.set_index('DT_FIM_EXERC').asfreq('Q')[['Revenue']]
    if method == 'ets':
        model = ExponentialSmoothing(
            np.log(data['Revenue']), trend=True, damped_trend=True, seasonal=4)
    elif method == 'arima':
        model = SARIMAX(
            np.log(data['Revenue']),
            order=(2, 1, 1), seasonal_order=(1, 0, 1, 4), trend='c')
    else:
        return {}
    results = model.fit()
    fcasts = np.exp(results.forecast(4*5))
    df = pd.concat([data, pd.DataFrame({'Forecast': fcasts})]).reset_index()
    fcasts = (
        pd.DataFrame({'Forecast': np.exp(results.forecast(4*5))})
        .reset_index()
        .rename(columns={'index': 'DT_FIM_EXERC'})
    )
    simulations = (
        np.exp(results.simulate(4*5, repetitions=100, anchor=data.shape[0]))
        .reset_index()
        .melt('index', value_name='Simulation')
        .drop(columns='variable_0')
        .rename(columns={'variable_1': 'iteration', 'index': 'DT_FIM_EXERC'})
    )
    df = pd.concat([
        data.reset_index().merge(fcasts, how='outer'),
        simulations
        ])
    df['iteration'] = df['iteration'].fillna('')
    fig = px.line(df,
        x='DT_FIM_EXERC', y=['Revenue', 'Simulation', 'Forecast'],
        line_group='iteration')
    return fig
