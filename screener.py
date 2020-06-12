import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output, State

from app import app, companies
from funcs import grid, calc_kpis


visible_cols = [
    'BTICKER', 'SETOR', 'Revenue', 'EBIT', 'NetIncome',
    'EBITMargin', 'NetMargin', 'ROIC', 'ROE', 'NetDebtToEBIT'
]
screener_table = dt.DataTable(
    id='screener_table',
    data=companies.to_dict('records'),
    columns=[{"name": i, "id": i} for i in companies.columns],
    hidden_columns=[i for i in companies.columns if i not in visible_cols],
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

