import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output, State

import plotly.express as px

from app import app, companies, fin_stmts
from funcs import grid, calc_kpis


accounts = pd.read_csv('accounts.csv')
bp_accounts = (
    accounts[accounts['CD_CONTA'].str[:1].isin(['1', '2'])]
    .melt('CD_CONTA')
    .dropna()
    ['value']
    .unique()
)


screener = companies.merge(
    fin_stmts
    .sort_values(['CD_CVM', 'DT_FIM_EXERC'])
    .groupby('CD_CVM').tail(4)
    .groupby('CD_CVM').agg({
        s: 'mean' if s in bp_accounts else 'sum'
        for s in fin_stmts.columns[2:]
    })
    .reset_index()
    ,
    on="CD_CVM", how='inner'
).sort_values('Revenue', ascending=False).pipe(calc_kpis, False)


filter_cols = [
    "MarketCap",
    "Revenue", "GrossProfit", "EBIT", "EBT", "NetIncome",
    "OperatingCashFlow", "FreeCashFlow",
    "GrossMargin", "EBITMargin", "NetMargin",
    "ROIC", "ROE",
    "DebtToEquity", "NetDebtToEBIT",
    "CurrentLiquidity", "GeneralLiquidity", "CashLiquidity",
]


layout = html.Div([
    html.H2('Screener'),
    grid([
        [
            html.Div([
                html.Label("Indicadores"),
                dcc.Dropdown(
                    id="screener_variables",
                    options=[{'label': s, 'value': s} for s in filter_cols],
                    value=["MarketCap", "Revenue", "EBITMargin", "NetMargin",
                           "ROIC", "ROE", "NetDebtToEBIT"],
                    multi=True,
                    persistence=True
                )
            ]),
            html.Div([
                html.Label("Ordem"),
                dcc.Dropdown(
                    id="screener_order",
                    options=[{'label': s, 'value': s} for s in filter_cols],
                    value="MarketCap",
                    persistence=True
                ),
                dbc.FormGroup([
                    dbc.Checkbox(
                        id="order_ascending", className="form-check-input",
                        checked=False, persistence=True
                    ),
                    dbc.Label(
                        'Crescente', html_for="order_ascending",
                        className="form-check-label"
                    )
                ], check=True)
                
            ])
        ],
        [
            dcc.Graph(id="screener_plot", style={'height': '80vh'})
        ]
    ])
    
])


@app.callback(
    Output("screener_plot", "figure"),
    [Input('screener_variables', 'value'),
     Input('screener_order', 'value'),
     Input('order_ascending', 'checked')]
)
def update_screener(variables, order, ascending):
    fig = px.bar(
        screener.sort_values(order, ascending=ascending).iloc[:40],
        y='BTICKER', x=variables, facet_col='variable', text='value',
        labels={"variable": "", "value": "", "BTICKER": ""}
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_xaxes(matches=None)
    fig.update_yaxes(autorange="reversed")
    fig.for_each_annotation(lambda a: a.update(
        text='<b>' + a.text.split("=")[-1] + "</b>"))
    fig.update_layout(showlegend=False)
    return fig



@app.callback(
    Output('url', 'pathname'),
    [Input('screener_plot', 'clickData')],
    [State("url", "pathname")]
)
def display_click_data(clickData, url):
    if url is None:
        return None
    else:
        ticker = clickData['points'][0]['y']
        return '/' + ticker
    return url
