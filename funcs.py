import numpy as np
import pandas as pd

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc


def spinner_graph(*args, **kwargs):
    return dbc.Spinner(dcc.Graph(*args, **kwargs))


def add_quarters(df):
    df['Quarter'] = pd.to_datetime(df['DT_FIM_EXERC']).dt.quarter.astype(str)
    for i in range(4):
        df[f'Q{i+1}'] = (df['Quarter'] == f"{i + 1}") * 1
    return df


def calc_kpis(df, quarterly=True):
    if 'DT_FIM_EXERC' in df.columns:
        df.sort_values('DT_FIM_EXERC', inplace=True)
        if quarterly:
            df = add_quarters(df)
    df['EBITDA'] = df['EBIT'] - df['DepreciationAmortization']
    if 'SEGMENTO' in df.columns:
        df['EBIT'] = np.where(
            df['SEGMENTO'] == 'Bancos', df['EBT'], df['EBIT']
        )
        df['EBITDA'] = df['EBIT']
    df['FreeCashFlow'] = df['OperatingCashFlow'] + df['InvestingCashFlow']
    df['Opex'] = df['Revenue'] - df['EBIT']
    df['ShareholderEquity'] = \
        df['StakeholderEquity'] - df['MinorityInterests'].fillna(0)
    df['NetDebt'] = df['Debt'].fillna(0) - df['Cash']
    df['InvestedCapital'] = df['ShareholderEquity'] + df['Debt'].fillna(0)
    df['GrossMargin'] = 100 * df['GrossProfit'] / df['Revenue']
    df['EBITMargin'] = 100 * df['EBIT'] / df['Revenue']
    df['EBITDAMargin'] = 100 * df['EBITDA'] / df['Revenue']
    df['NetMargin'] = 100 * df['NetIncome'] / df['Revenue']
    df['DebtToEquity'] = np.where(df['ShareholderEquity'] < 0, np.NaN,
                                  df['Debt'] / df['ShareholderEquity'])
    df['DebtToCapital'] = df['Debt'] / df['InvestedCapital']

    df['CurrentLiquidity'] = df['CurrentAssets'] / df['CurrentLiabilities']
    df['GeneralLiquidity'] = (df['CurrentAssets'] + df['LongTermAssets']) / \
        (df['CurrentLiabilities'] + df['LongTermLiabilities'])
    df['CashLiquidity'] = df['Cash'] / df['CurrentLiabilities']

    if 'MarketCap' in df:
        df['FirmValue'] = df['MarketCap'] + df['NetDebt']
        df['PE'] = np.where(
            df['NetIncome'] <= 0, np.NaN, df['MarketCap'] / df['NetIncome'])
        df['PB'] = np.where(
            df['ShareholderEquity'] <= 0, np.NaN,
            df['MarketCap'] / df['ShareholderEquity'])
        df['EV2EBIT'] = np.where(
            df['EBIT'] <= 0, np.NaN, df['FirmValue'] / df['EBIT'])
        df['EV2EBITDA'] = np.where(
            df['EBITDA'] <= 0, np.NaN, df['FirmValue'] / df['EBITDA'])
        df['EV2AdjEBITDA'] = np.where(
            df['AdjustedEBITDA'] <= 0, np.NaN,
            df['FirmValue'] / df['AdjustedEBITDA'])
        df['EV2FCFF'] = np.where(
            df['FreeCashFlow'] <= 0, np.NaN,
            df['FirmValue'] / df['FreeCashFlow'])

    if quarterly:
        df['RevenueGrowth'] = 100 * (df['Revenue'] / df['Revenue'].shift(4) -1)
        df['LTM_NetIncome'] = df['NetIncome'].rolling(4).sum()
        df['LTM_Revenue'] = df['Revenue'].rolling(4).sum()
        df['LTM_COGS'] = df['COGS'].rolling(4).sum()
        df['LTM_EBIT'] = df['EBIT'].rolling(4).sum()
        df['LTM_ShareholderEquity'] = df['ShareholderEquity'].rolling(4).mean()
        df['LTM_InvestedCapital'] = df['InvestedCapital'].rolling(4).mean()
        df['ROE'] = 100 * df['LTM_NetIncome'] / df['LTM_ShareholderEquity']
        df['ROIC'] = 100 * df['LTM_EBIT'] / df['LTM_InvestedCapital']
        df['NetDebtToEBIT'] = np.where(df['LTM_EBIT'] < 0, np.NaN,
                                   df['NetDebt'] / df['LTM_EBIT'])
        df['DaysSalesOutstanding'] = \
            df['AccountsReceivable'] / df['LTM_Revenue'] * 365
        df['DaysInventoriesOutstanding'] = \
            - df['Inventories'] / df['LTM_COGS'] * 365
        df['DaysPayablesOutstanding'] = \
            - df['AccountsPayable'] / df['LTM_COGS'] * 365
        df['CashConversionCycle'] = df['DaysSalesOutstanding'] + \
            df['DaysInventoriesOutstanding'] - \
            df['DaysPayablesOutstanding']
    else:
        df['ROE'] = 100 * df['NetIncome'] / df['ShareholderEquity']
        df['ROIC'] = 100 * df['EBIT'] / df['InvestedCapital']
        df['NetDebtToEBIT'] = np.where(df['EBIT'] < 0, np.NaN,
                                   df['NetDebt'] / df['EBIT'])
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
