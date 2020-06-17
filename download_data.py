import numpy as np
import pandas as pd
from data_funcs import get_cvm_all, get_companies, get_pib, get_ipca, get_usd

# Macroeconomic data
macro = (
    get_pib()
    .merge(get_ipca())
    .merge(get_usd())
)
macro.to_csv('data/macro.csv')


# Financial statements
accounts = pd.read_csv('accounts.csv')

companies = get_companies()
companies.to_csv('data/companies.csv', index=False)

fin_stmts = get_cvm_all(
    range(2010, 2021),
    accounts=accounts['CD_CONTA'].unique(),
    companies=companies['CD_CVM'].unique())
fin_stmts.to_csv('data/fin_stmts_long.csv', index=False)


accounts = accounts.melt('CD_CONTA', var_name='BANCO', value_name='VARNAME')
accounts = accounts[accounts['VARNAME'].isna() == False]

bancos = companies['CD_CVM'][companies['SEGMENTO'] == 'Bancos']
fin_stmts['BANCO'] = \
    np.where(fin_stmts['CD_CVM'].isin(bancos), 'BANCO', 'OUTRA')

fin_stmts_wide = (
    fin_stmts
    .merge(accounts, on=['BANCO', 'CD_CONTA'], how='inner')
    .groupby(['CD_CVM', 'DT_FIM_EXERC', 'VARNAME'])
    ['VL_CONTA'].sum().reset_index()
    .pivot_table(index=['CD_CVM', 'DT_FIM_EXERC'],
                 columns='VARNAME', values='VL_CONTA')
    .reset_index()
)
fin_stmts_wide.to_csv('data/fin_stmts_wide.csv', index=False)
