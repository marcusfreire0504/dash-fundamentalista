import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px

from app import app, companies, fin_stmts, colorscheme
from funcs import grid, calc_kpis, spinner_graph


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
    "MarketCap", "NetDebt", "FirmValue",
    "Revenue", "GrossProfit", "EBITDA", "EBIT", "EBT", "NetIncome",
    "OperatingCashFlow", "FreeCashFlow",
    "GrossMargin", "EBITDAMargin", "EBITMargin", "NetMargin",
    "ROIC", "ROE",
    "PE", "EV2EBIT", "EV2EBITDA", "PB",
    "DebtToEquity", "NetDebtToEBIT",
    "CurrentLiquidity", "GeneralLiquidity", "CashLiquidity",
]


layout = html.Div([
    html.H2('Screener'),
    dbc.Row([
        dbc.Col([
            html.Label("Indicadores"),
            dcc.Dropdown(
                id="screener_variables",
                options=[{'label': s, 'value': s} for s in filter_cols],
                value=["MarketCap", "EBITMargin", "NetMargin",
                        "ROIC", "ROE", "NetDebtToEBIT", "EV2EBIT", "PE"],
                multi=True,
                persistence=True,
                clearable=False
            )
        ], width=8),
        dbc.Col([
            html.Label("Ordem"),
            dcc.Dropdown(
                id="screener_order",
                options=[{'label': s, 'value': s} for s in filter_cols],
                value="MarketCap",
                clearable=False,
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
    ]),
    dbc.Tabs([
        dbc.Tab([
            spinner_graph(id="screener_plot", style={'height': '80vh'})
        ], label="Barras"),
        dbc.Tab([
            spinner_graph(id="screener_scatter", style={'height': '80vh'})
        ], label="Dispersão")
    ])
])


@app.callback(
    [Output("screener_plot", "figure"),
     Output("screener_scatter", "figure")],
    [Input('screener_variables', 'value'),
     Input('screener_order', 'value'),
     Input('order_ascending', 'checked')]
)
def update_screener(variables, order, ascending):
    df = screener.sort_values(order, ascending=ascending).head(40)
    fig = px.bar(
        df,
        y='BTICKER', x=variables, facet_col='variable', text='value',
        color_discrete_sequence=[colorscheme[0]],
        labels={"variable": "", "value": "", "BTICKER": ""}
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_xaxes(matches=None)
    fig.update_yaxes(autorange="reversed")
    fig.for_each_annotation(lambda a: a.update(
        text='<b>' + a.text.split("=")[-1] + "</b>"))
    fig.update_layout(showlegend=False)

    scatter = px.scatter(
        df,
        size=variables[0],
        x=variables[1],
        y=variables[2],
        text="BTICKER",
        color="SETOR"
    )
    scatter.update_traces(textposition='top center')
    scatter.update_layout(legend=dict(orientation='v'))

    return fig, scatter



@app.callback(
    Output('url', 'pathname'),
    [Input('screener_plot', 'clickData')],
    [State("url", "pathname")]
)
def display_click_data(clickData, url):
    if clickData is None:
        raise PreventUpdate
    else:
        ticker = clickData['points'][0]['y']
        return '/' + ticker
    return url
