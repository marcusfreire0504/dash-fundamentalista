import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output, State

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
    .pipe(calc_kpis, False),
    on="CD_CVM", how='inner'
).sort_values('Revenue', ascending=False)


visible_cols = [
    'BTICKER', 'SETOR', 'Revenue', 'EBIT', 'NetIncome',
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


layout = html.Div([
    html.H2('Screener'),
    grid([[screener_table]])

])

