import numpy as np

import dash_html_components as html
import dash_bootstrap_components as dbc


def calc_kpis(df):
    df['ShareholderEquity'] = \
        df['StakeholderEquity'] - df['MinorityInterests'].fillna(0)
    df['NetDebt'] = df['Debt'].fillna(0) - df['Cash']
    df['InvestedCapital'] = df['ShareholderEquity'] + df['Debt'].fillna(0)
    
    df['ROE'] = 100 * df['NetIncome'] / df['ShareholderEquity']
    df['ROIC'] = 100 * df['EBIT'] / df['InvestedCapital']
    df['GrossMargin'] = 100 * df['GrossProfit'] / df['Revenue']
    df['EBITMargin'] = 100 * df['EBIT'] / df['Revenue']
    df['NetMargin'] = 100 * df['NetIncome'] / df['Revenue']
    df['NetDebtToEBIT'] = np.where(df['EBIT'] < 0, np.NaN,
                                   100 * df['NetDebt'] / df['EBIT'])
    df['DebtToEquity'] = np.where(df['ShareholderEquity'] < 0, np.NaN,
                                  df['Debt'] / df['ShareholderEquity'])
    df['DebtToCapital'] = df['Debt'] / df['InvestedCapital']
    return df


#
def grid(rows):
    return html.Div([
        dbc.Row([
            dbc.Col(col, width=12/len(row))
            for col in row
        ])
        for row in rows
    ])
