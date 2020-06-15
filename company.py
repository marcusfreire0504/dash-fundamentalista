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

simulation_scheme = [colorscheme[0], 'rgba(180,180,180,0.2)', '#0f0f0f']


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
        dcc.Store(id='models_store', data={}),
        html.H2(company_name),
        html.Ol(sectors, className='breadcrumb',
            style={'background': 'none'}),
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
    [Output('rev_forecast_store', 'data'),
     Output('models_store', 'data')],
    [Input('stmts_store', 'data'),
     Input('rev_forecast_method', 'value')]

)
def update_revenue_forecast(historicals, method):
    historicals = pd.DataFrame(historicals)
    historicals['DT_FIM_EXERC'] = pd.to_datetime(historicals['DT_FIM_EXERC'])
    models = {}

    # Revenue time series model
    data = historicals.set_index('DT_FIM_EXERC').asfreq('Q')[['Revenue']]
    if method == 'ets':
        rev_model = ExponentialSmoothing(
            np.log(data['Revenue']), trend=True, damped_trend=True, seasonal=4)
    elif method == 'arima':
        rev_model = SARIMAX(
            np.log(data['Revenue']),
            order=(2, 1, 1), seasonal_order=(1, 0, 1, 4), trend='c')
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
    nfolds = round(historicals.shape[0] / (4 * foldsize)) - 1
    cv_errors = []
    for fold in range(nfolds, 0, -1):
        train_subset = historicals.iloc[:-(fold+2)*(4*foldsize)]
        valid_subset = historicals.iloc[-(fold+2)*(4*foldsize):-(fold+1)*(4*foldsize)]
        if train_subset.shape[0] < 16:
            continue
        fcasts = (
            rev_model.clone(np.log(train_subset['Revenue']))
            .fit().forecast(valid_subset.shape[0])
        )
        cv_errors = np.append(
            cv_errors, fcasts - np.log(valid_subset['Revenue'])
        )
    models['revenue']['diag']['CV RMSE'] = np.sqrt(np.mean(
        (cv_errors) ** 2
    ))
    models['revenue']['diag']['CV MAE'] = np.mean(
        np.abs(cv_errors)
    )

    # Generate simulated forecasts
    nsim = 100
    horiz = 5
    forecasts = (
        pd.DataFrame({
                'Revenue': np.exp(rev_results.forecast(horiz * 4)),
                'group': 'forecast', 'variable_1': ''
            })
        .reset_index()
    )
    simulations = (
        rev_results.simulate(
            4 * horiz, repetitions=nsim, anchor=data.shape[0]
        )
        .pipe(np.exp)
        .reset_index()
        .melt('index', value_name='Revenue')
        .drop(columns='variable_0')
        .assign(group='simulation')
    )
    simulations = (
        pd.concat([simulations, forecasts])
        .reset_index(drop=True)
        .rename(columns={'variable_1': 'iteration', 'index': 'DT_FIM_EXERC'})
        .pipe(add_quarters)
        .assign(
            RevenueGrowth=lambda x: 100 * (x['Revenue'] /
                x.groupby('iteration')['Revenue'].shift(4) - 1)
        )
    )
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
        color_discrete_sequence=simulation_scheme)
    fig.update_yaxes(matches=None)
    text = "<br>".join([
        f"<b>{k}:</b> {round(v, 4)}" for k, v in model['diag'].items()
    ])
    fig.add_annotation(x=0, y=1, xref='paper', yref='paper', showarrow=False,
        text=text)
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
    Output('opex_forecast_plot', 'figure'),
    [Input('rev_forecast_store', 'data')]
)
def plot_opex_forecast(forecasts):
    df = pd.DataFrame(forecasts)
    fig = px.line(df, x='DT_FIM_EXERC', y=['Opex', 'EBIT', 'EBITMargin'],
        color='group', color_discrete_sequence=simulation_scheme,
        facet_col='variable', facet_col_wrap=1, line_group='iteration')
    fig.update_yaxes(matches=None)

    return fig
