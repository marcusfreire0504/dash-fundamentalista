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
from statsmodels.regression.quantile_regression import QuantReg

from app import app, companies, fin_stmts
from funcs import grid, calc_kpis, add_quarters


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
        dcc.Store(id='rev_forecast_store', data={}),
        dcc.Store(id='ebit_forecast_store', data={}),
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
            ], label="Receita"),
            dbc.Tab([
                grid([[
                    dcc.Graph(id='opex_scatter', style={'height': '80vh'}),
                    dcc.Graph(id='opex_forecast_plot', style={'height': '80vh'})
                ]])
            ], label='OPEX')
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
    Output('rev_forecast_store', 'data'),
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
    simulations = (
        np.exp(results.simulate(4*5, repetitions=100, anchor=data.shape[0]))
        .reset_index()
        .melt('index', value_name='Revenue')
        .drop(columns='variable_0')
        .rename(columns={'variable_1': 'iteration', 'index': 'DT_FIM_EXERC'})
    )
    simulations['RevenueGrowth'] = 100 * (simulations['Revenue'] /
            simulations.groupby('iteration')['Revenue'].shift(4) - 1)
    simulations = add_quarters(simulations)
    return simulations.to_dict('records')


@app.callback(
    Output('rev_forecast_plot', 'figure'),
    [Input('stmts_store', 'data'),
     Input('rev_forecast_store', 'data')]
)
def plot_revenue_forecast(historicals, forecasts):
    historicals = pd.DataFrame(historicals)
    forecasts = pd.DataFrame(forecasts)
    df = pd.concat([historicals, forecasts])
    df['iteration'] = df['iteration'].fillna('')
    fig = px.line(df,
        x='DT_FIM_EXERC', y=['Revenue', 'RevenueGrowth'],
        line_group='iteration',
        facet_col='variable', facet_col_wrap=1)
    fig.update_yaxes(matches=None)
    return fig


@app.callback(
    Output('opex_scatter', 'figure'),
    [Input('stmts_store', 'data')]
)
def plot_opex_scatter(data):
    df = pd.DataFrame(data)
    fig = px.scatter(
        df, x='Revenue', y='Opex', color='Quarter', size='EBITMargin')
    return fig


@app.callback(
    Output('ebit_forecast_store', 'data'),
    [Input('stmts_store', 'data'),
     Input('rev_forecast_store', 'data')]
)
def update_ebit_forecast(historicals, forecasts):
    historicals = pd.DataFrame(historicals)
    forecasts = pd.DataFrame(forecasts)
    historicals['const'] = 1
    historicals['logRevenue'] = np.log(historicals['Revenue'])

    exog = historicals[['logRevenue', 'Q2', 'Q3', 'Q4']]
    exog = sm.add_constant(exog)
    
    model = QuantReg(np.log(historicals['Opex']), exog)
    results = model.fit(q=0.5)
    coefs = results.params
    rmse = np.mean(results.resid ** 2) ** .5

    forecasts['Opex'] = np.exp(
        coefs[0] + coefs[1] * np.log(forecasts['Revenue']) +
        coefs[2] * forecasts['Q2'] + coefs[3] * forecasts['Q3'] +
        coefs[4] * forecasts['Q4'] +
        np.random.normal(0, rmse, forecasts.shape[0])
    )
    forecasts['EBIT'] = forecasts['Revenue'] - forecasts['Opex']
    forecasts['EBITMargin'] = 100 * forecasts['EBIT'] / forecasts['Revenue']

    return forecasts.to_dict('records')



@app.callback(
    Output('opex_forecast_plot', 'figure'),
    [Input('stmts_store', 'data'),
     Input('ebit_forecast_store', 'data')]
)
def plot_opex_forecast(historicals, forecasts):
    historicals = pd.DataFrame(historicals)
    forecasts = pd.DataFrame(forecasts)

    cols = [s for s in forecasts.columns if s in historicals.columns]
    df = pd.concat([historicals[cols], forecasts])
    df['iteration'] = df['iteration'].fillna('')

    fig = px.line(df, x='DT_FIM_EXERC', y=['Opex', 'EBIT', 'EBITMargin'],
        facet_col='variable', facet_col_wrap=1, line_group='iteration')
    fig.update_yaxes(matches=None)

    return fig
