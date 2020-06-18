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

from app import app, companies, fin_stmts, colorscheme
from funcs import grid, calc_kpis, add_quarters
from data_funcs import get_focus, get_quotes

simulation_scheme = [colorscheme[0], 'rgba(180,180,180,0.2)', '#0f0f0f']

macro = pd.read_csv("data/macro.csv")
macro['USD'] = macro['USD_AVG']
focus = get_focus(macro)


def layout(ticker):
    arima_marks = {i: str(i) for i in range(4)}
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
    data = data.merge(macro, on="DT_FIM_EXERC")
    #
    quotes = pd.read_csv('data/tickers.txt', names=['ticker'])
    quotes = quotes[quotes['ticker'].str[:4] == ticker]['ticker'].values
    quotes = get_quotes(quotes)
    quotes['tipo'] = quotes['ticker'].str[4:]
    quotes['qtde'] = np.where(
        quotes['tipo'] == '3', row['QTDE_ON'],
        row['QTDE_PN'] / np.sum(quotes['tipo'] != '3')
    )
    quotes['MarketCap'] = quotes['qtde'] * quotes['cotacao'] / 1000000000
    mktcap = quotes['MarketCap'].sum()
    
    #
    cards = [
        dbc.Card(
            dbc.CardBody([
                html.H5(quotes['ticker'].iloc[i], className="card-title"),
                html.H1(f'R$ {quotes["cotacao"].iloc[i]}')
            ])
        )
        for i in range(quotes.shape[0])
    ]
    cards.append(
        dbc.Card(
            dbc.CardBody([
                html.H5("Market-cap"),
                html.H1(f"R$ {round(mktcap, 1)} bi")
            ])
        )
    )
    #
    return html.Div([
        dcc.Store(id='stmts_store', data=data.to_dict('records')),
        dcc.Store(id='rev_forecast_store', data={}),
        dcc.Store(id='models_store', data={}),
        html.H2(company_name),
        html.Ol(sectors, className='breadcrumb',
            style={'background': 'none'}),
        dbc.CardGroup(cards),
        dbc.Tabs([
            dbc.Tab([
                grid([
                    [
                        dcc.Graph(id='ov_revenue_plot', style={'height': '40vh'}),
                        dcc.Graph(id='ov_profit_plot', style={'height': '40vh'})
                    ],
                    [
                        dcc.Graph(id='ov_margins_plot', style={'height': '40vh'}),
                        dcc.Graph(id='ov_returns_plot', style={'height': '40vh'})
                    ]
                ])
            ], label="Visão Geral"),
            dbc.Tab([
                grid([
                    [
                        dcc.Graph(id='workingk_plot', style={'height': '40vh'}),
                        dcc.Graph(id='liquid_plot', style={'height': '40vh'}),
                    ],
                    [
                        dcc.Graph(id='ov_debt_plot', style={'height': '40vh'}),
                        dcc.Graph(id='ov_debt2_plot', style={'height': '40vh'})
                    ]
                ])
            ], label="Capital"),
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Label('Indexador'),
                        dbc.RadioItems(
                            id="forecast_index",
                            value="ipca",
                            inline=True,
                            options=[
                                {'value': '', 'label': 'Nenhum'},
                                {'value': 'ipca', 'label': 'IPCA'},
                                {'value': 'usd', 'label': 'USD'}
                            ],
                            persistence=ticker
                        ),
                        html.Div([
                            html.Label('Cenário Focus'),
                            dbc.RadioItems(
                                id="focus_scenario",
                                value="Mediana",
                                inline=True,
                                options=[
                                    {'value': s, 'label': s}
                                    for s in focus['scenario'].unique()
                                ],
                                persistence=ticker
                            )
                        ], id="focus_scenario_div", style={"display": "block"}),
                        html.Label('Método'),
                        dbc.RadioItems(
                            id='rev_forecast_method',
                            value='ets',
                            inline=True,
                            options=[
                                {'value': 'ets', 'label': 'Alisamento exponencial'},
                                {'value': 'arima', 'label': 'ARIMA'}
                            ],
                            persistence=ticker
                        ),
                        html.Div([
                            html.Label('Coef. Autoregressivos (p)'),
                            dcc.Slider(id="arima_p", min=0, max=3, value=2,
                                marks=arima_marks,
                                persistence=ticker),
                            html.Label('Ordem de integração (d)'),
                            dcc.Slider(id="arima_d", min=0, max=3, value=1,
                                marks=arima_marks,
                                persistence=ticker),
                            html.Label('Coef. Média Móvel (q)'),
                            dcc.Slider(id="arima_q", min=0, max=3, value=1,
                                marks=arima_marks,
                                persistence=ticker),
                            html.Label('Coef. AR sazonal (P)'),
                            dcc.Slider(id="arima_P", min=0, max=3, value=1,
                                marks=arima_marks,
                                persistence=ticker),
                            html.Label('Ordem de integração sazonal (D)'),
                            dcc.Slider(id="arima_D", min=0, max=3, value=0,
                                marks=arima_marks,
                                persistence=ticker),
                            html.Label('Coef. Média Móvel sazonal (Q)'),
                            dcc.Slider(id="arima_Q", min=0, max=3, value=1,
                                marks=arima_marks,
                                persistence=ticker)
                        ], id="arima_params_div", style={"display": "none"})
                    ], width=3, className="sidebar"),
                    dbc.Col([
                        dcc.Graph("rev_forecast_plot", style={'height': '80vh'})
                    ], width=9)
                ])
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
     Output('ov_debt2_plot', 'figure'),
     Output('workingk_plot', 'figure'),
     Output('liquid_plot', 'figure')],
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
    workingk_fig = px.line(
        df,
        x='DT_FIM_EXERC',
        y=['DaysSalesOutstanding', 'DaysInventoriesOutstanding',
           'DaysPayablesOutstanding', 'CashConversionCycle'],
        color='variable',
        title='Capital de Giro', labels=labs
    )
    liquid_fig = px.line(
        df,
        x='DT_FIM_EXERC',
        y=[f'{s}Liquidity' for s in ['Current', 'General', 'Cash']],
        color='variable',
        title='Liquidez', labels=labs
    )
    
    return revenue_fig, profit_fig, margins_fig, returns_fig, \
        debt_fig, debt2_fig, workingk_fig, liquid_fig


@app.callback(
    [Output('arima_params_div', 'style'),
     Output('focus_scenario_div', 'style')],
    [Input('rev_forecast_method', 'value'),
     Input('forecast_index', 'value')]
)
def update_arima_params_visible(method, fcast_index):
    if method == "arima":
        arima_display = 'block'
    else:
        arima_display = 'none'
    if fcast_index == '':
        focus_display = 'none'
    else:
        focus_display = 'block'
    return {"display": arima_display}, {"display": focus_display}


@app.callback(
    [Output('rev_forecast_store', 'data'),
     Output('models_store', 'data')],
    [Input('stmts_store', 'data'),
     Input('rev_forecast_method', 'value'),
     Input('forecast_index', 'value'),
     Input('focus_scenario', 'value'),
     Input('arima_p', 'value'),
     Input('arima_d', 'value'),
     Input('arima_q', 'value'),
     Input('arima_P', 'value'),
     Input('arima_D', 'value'),
     Input('arima_Q', 'value')]
)
def update_revenue_forecast(historicals, method, fcast_index, focus_scenario,
                            p, d, q, P, D, Q):
    historicals = pd.DataFrame(historicals)
    historicals['DT_FIM_EXERC'] = pd.to_datetime(historicals['DT_FIM_EXERC'])
    models = {}

    # Revenue time series model
    data = historicals.set_index('DT_FIM_EXERC').asfreq('Q')
    y = data['Revenue']
    # Transform
    if fcast_index != '':
        idx = data[fcast_index.upper()]
        y = y / idx * idx.iloc[-1]
    y = np.log(y)
    
    # Create forecast model
    if method == 'ets':
        rev_model = ExponentialSmoothing(
            y, trend=True, damped_trend=True, seasonal=4)
    elif method == 'arima':
        rev_model = SARIMAX(
            y,
            order=(p, d, q), seasonal_order=(P, D, Q, 4), trend='c')
    else:
        return {}
    rev_results = rev_model.fit()
    models['revenue'] = {
        'Params': rev_results.params,
        'diag': {
            'In-sample RMSE': np.sqrt(rev_results.mse),
            'In-sample MAE': rev_results.mae,
            'Ljung-Box': rev_results.test_serial_correlation('ljungbox')[0, 0, -1],
            'log-Likelihood': rev_results.llf,
            'AICc': rev_results.aicc,
            'BIC': rev_results.bic
        }
    }
    # Cross validation
    foldsize = 1
    nfolds = round(y.shape[0] / (4 * foldsize)) - 1
    cv_errors = []
    for fold in range(nfolds, 0, -1):
        train_subset = y.iloc[:-(fold+2)*(4*foldsize)]
        valid_subset = y.iloc[-(fold+2)*(4*foldsize):-(fold+1)*(4*foldsize)]
        if train_subset.shape[0] < 16:
            continue
        fcasts = (
            rev_model.clone(np.log(train_subset))
            .fit().forecast(valid_subset.shape[0])
        )
        cv_errors = np.append(
            cv_errors, fcasts - np.log(valid_subset)
        )
    if len(cv_errors) > 4:
        models['revenue']['diag']['CV RMSE'] = np.sqrt(np.mean(
            np.array(cv_errors) ** 2
        ))
        models['revenue']['diag']['CV MAE'] = np.mean(
            np.abs(cv_errors)
        )

    # Generate simulated forecasts
    nsim = 100
    horiz = int(np.sum(focus['scenario'] == focus_scenario))
    forecasts = (
        pd.DataFrame({
                'y': rev_results.forecast(horiz),
                'group': 'forecast', 'variable_1': ''
            })
        .reset_index()
    )
    simulations = (
        rev_results.simulate(
            horiz, repetitions=nsim, anchor=data.shape[0]
        )
        .reset_index()
        .melt('index', value_name='y')
        .drop(columns='variable_0')
        .assign(group='simulation')
    )

    simulations = (
        pd.concat([simulations, forecasts])
        .reset_index(drop=True)
        .rename(columns={'variable_1': 'iteration', 'index': 'DT_FIM_EXERC'})
        .pipe(add_quarters)
    )
    simulations['Revenue'] = np.exp(simulations['y'])
    if fcast_index != '':
        simulations = simulations.merge(
            focus[['DT_FIM_EXERC', fcast_index.upper()]]
            [focus['scenario'] == focus_scenario],
            on="DT_FIM_EXERC", how="left"
        )
        simulations['Revenue'] = simulations['Revenue'] \
            * simulations[fcast_index.upper()] \
            / data[fcast_index.upper()].iloc[-1]
            

    simulations['RevenueGrowth'] = 100 * (simulations['Revenue'] /
        simulations.groupby('iteration')['Revenue'].shift(4) - 1)
    simulations.loc[simulations['RevenueGrowth'].isna(), 'RevenueGrowth'] = \
        np.reshape(
            100 * (
                np.reshape(
                    simulations['Revenue'][simulations['RevenueGrowth'].isna()].values,
                    (nsim + 1, 4)) / 
                historicals['Revenue'].tail(4).values - 1
                ),
            ((nsim + 1) * 4)
        )

    # Expenses regression model
    historicals['logRevenue'] = np.log(historicals['Revenue'])
    exog = historicals[['logRevenue', 'Q1', 'Q2', 'Q3', 'Q4']]
    
    opex_model = QuantReg(np.log(historicals['Opex']), exog)
    opex_results = opex_model.fit(q=0.5)
    opex_coefs = opex_results.params
    rmse = np.mean(opex_results.resid ** 2) ** .5

    models['opex'] = {
        'Params': opex_results.params,
        'diag': {
            'In-sample RMSE': np.sqrt(np.mean(opex_results.resid)**2),
            'In-sample MAE': np.mean(np.abs(opex_results.resid)),
            #'Ljung-Box': opex_results.test_serial_correlation('ljungbox')[0, 0, -1],
            #'log-Likelihood': opex_results.llf,
            #'AICc': opex_results.aicc,
            #'BIC': opex_results.bic
        }
    }

    # Simulations
    simulations['Opex'] = np.exp(
        opex_coefs[0] * np.log(simulations['Revenue']) +
        opex_coefs[1] * simulations['Q1'] + opex_coefs[2] * simulations['Q2'] +
        opex_coefs[3] * simulations['Q3'] + opex_coefs[4] * simulations['Q4'] +
        np.random.normal(0, rmse, simulations.shape[0]) *
        (simulations['group'] == 'simulation')
    )
    simulations['EBIT'] = simulations['Revenue'] - simulations['Opex']
    simulations['EBITMargin'] = 100 * simulations['EBIT'] / simulations['Revenue']
    simulations['Taxes'] = simulations['EBIT'] * .34
    simulations['NOPAT'] = simulations['EBIT'] - simulations['Taxes']

    simulations = pd.concat([
        historicals.assign(group='historicals', iteration=''),
        simulations
    ])

    return simulations.to_dict('records'), models


@app.callback(
    Output('rev_forecast_plot', 'figure'),
    [Input('rev_forecast_store', 'data'),
     Input('models_store', 'data')]
)
def plot_revenue_forecast(forecasts, models):
    model = models['revenue']
    df = pd.DataFrame(forecasts)
    fig = px.line(df,
        x='DT_FIM_EXERC', y=['Revenue', 'RevenueGrowth'],
        line_group='iteration', color='group',
        facet_col='variable', facet_col_wrap=1,
        color_discrete_sequence=simulation_scheme,
        labels={'DT_FIM_EXERC': '', 'value': '', 'variable': ''})
    fig.update_yaxes(matches=None)
    text = "<br>".join([
        f"<b>{k}:</b> {round(v, 4)}" for k, v in model['diag'].items()
    ])
    fig.add_annotation(x=0, y=1, xref='paper', yref='paper', showarrow=False,
        text=text, align='left')
    fig.update_layout(showlegend=False)
    return fig


@app.callback(
    Output('opex_scatter', 'figure'),
    [Input('stmts_store', 'data'),
     Input('models_store', 'data')]
)
def plot_opex_scatter(data, models):
    df = pd.DataFrame(data)
    model = models['opex']
    fig = px.scatter(
        df, x='Revenue', y='Opex', color='Quarter')
    text = "<br>".join([
        f"<b>{k}:</b> {round(v, 4)}" for k, v in model['diag'].items()
    ])
    fig.add_annotation(x=0, y=1, xref='paper', yref='paper', showarrow=False,
        text=text, align='left')
    return fig


@app.callback(
    Output('opex_forecast_plot', 'figure'),
    [Input('rev_forecast_store', 'data')]
)
def plot_opex_forecast(forecasts):
    df = pd.DataFrame(forecasts)
    fig = px.line(df, x='DT_FIM_EXERC', y=['Opex', 'EBIT', 'EBITMargin'],
        color='group', color_discrete_sequence=simulation_scheme,
        facet_col='variable', facet_col_wrap=1, line_group='iteration',
        labels={'DT_FIM_EXERC': '', 'value': '', 'variable': ''})
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False)

    return fig
