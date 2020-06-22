import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from app import app, companies
import screener
import company


server = app.server

#
navbar = dbc.NavbarSimple(
    children=[],
    brand=app.title,
    brand_href="/",
    color='dark',
    dark=True
)


# LAYOUT
app.layout = html.Div([
    dcc.Location('url', refresh=False),
    navbar,
    html.Div(id='page_content', className='container-fluid'),
    html.Footer([
        html.Div([
            'Este aplicativo tem objetivo exclusivamente educacional e ' + \
            'todos os dados possuem caráter informativo. Não nos ' + \
            'responsabilizamos pelas decisões e caminhos tomados tomados ' + \
            'pelo usuário a partir da análise das informações aqui ' + \
            'disponibilizadas.'
        ], className='container')
    ], className='footer text-muted')
])


@app.callback(
    Output("page_content", "children"),
    [Input('url', 'pathname')]
)
def update_url(url):
    if url is None or len(url) < 4:
        ticker = ''
    else:
        ticker = url[1:5].upper()
        if companies['BTICKER'].str[:4].isin([ticker]).sum() == 0:
            ticker = ''
    if ticker == '':
        return screener.layout
    else:
        return company.layout(ticker)


#
if __name__ == "__main__":
    app.run_server(debug=True)
