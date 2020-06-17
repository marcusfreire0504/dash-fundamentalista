import os
import io
import numpy as np
import pandas as pd
import requests
import urllib.request as ur
from bs4 import BeautifulSoup
from zipfile import ZipFile




def get_sectors():
    url = 'http://bvmf.bmfbovespa.com.br/cias-listadas/empresas-listadas/' + \
        'BuscaEmpresaListada.aspx?opcao=1&indiceAba=1&Idioma=pt-br'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    url = soup.find("a", string="Download").get('href')

    # Unzip
    filehandle, _ = ur.urlretrieve(url)
    with ZipFile(filehandle, 'r') as zf:
        fn = zf.namelist()[0]
        df = pd.read_excel(
            io.BytesIO(zf.read(fn)),
            skiprows=7, skipfooter=18,
            names=['SETOR', 'SUBSETOR', 'NM_PREGAO', 'BTICKER', 'CD_GOVERN']
        )
    df['SEGMENTO'] = np.where(
        df['BTICKER'].isnull(), df['NM_PREGAO'], np.NaN)
    for col in ['SETOR', 'SUBSETOR', 'SEGMENTO']:
        df[col] = df[col].fillna(method='ffill')
    df['CD_GOVERN'] = df['CD_GOVERN'].fillna('')
    df = df.dropna(subset=['BTICKER'])
    df = df[df['BTICKER'] != 'CÓDIGO']
    df = df[df['SUBSETOR'] != 'SUBSETOR']
    df['NM_PREGAO'] = df['NM_PREGAO'].str.strip()
    df = df.reset_index(drop=True)
    return df[['BTICKER', 'NM_PREGAO', 'SETOR', 'SUBSETOR', 'SEGMENTO', 'CD_GOVERN']]


def get_listed_codes():
    url = 'http://bvmf.bmfbovespa.com.br/cias-listadas/empresas-listadas/' + \
           'BuscaEmpresaListada.aspx?idioma=pt-br'
    r = requests.post(url, {'__EVENTTARGET': 'ctl00:contentPlaceHolderConteudo:BuscaNomeEmpresa1:btnTodas'})
    soup = BeautifulSoup(r.text, 'html.parser')
    anchors = soup.find_all('a')
    df = pd.DataFrame({
        'CD_CVM': [a['href'] for a in anchors],
        'NM_PREGAO': [a.text for a in anchors]
    })
    df['NM_PREGAO'] = df['NM_PREGAO'].str.strip()
    df = df[df['CD_CVM'].str[:28] == 'ResumoEmpresaPrincipal.aspx?']
    df['CD_CVM'] = df['CD_CVM'].str.split('=', expand=True).loc[:,1]
    df = df[df.index % 2 == 0].reset_index(drop=True)
    return df


def get_index_composition(index_name='IBRA'):
    url = 'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?' + \
        f'Indice={index_name}'
    acoes = pd.read_html(url)[0]
    acoes.columns = ['TICKER', 'NM_PREGAO', 'TIPO', 'QTDE', 'PESO']
    acoes['NM_PREGAO'] = acoes['NM_PREGAO'].str.strip()
    acoes['PESO'] = acoes['PESO'] / 1000
    acoes['TIPO'] = acoes['TIPO'].str.split(' ', expand=True).loc[:,0]
    acoes = acoes[acoes['PESO'] != 100]
    acoes = acoes.sort_values('PESO', ascending=False)
    return acoes.reset_index(drop=True)


def get_num_shares():
    numshares = pd.read_html(
        'http://bvmf.bmfbovespa.com.br/CapitalSocial/',
        decimal=',', thousands='.')[0]
    numshares = numshares[['Código', 'Qtde Ações Ordinárias',
                           'Qtde Ações Preferenciais']]
    numshares.columns = ['BTICKER', 'QTDE_ON', 'QTDE_PN']
    numshares = numshares.groupby('BTICKER').sum().reset_index()
    return numshares.reset_index(drop=True)


def cache_data(fn, fun, *args, **kwargs):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    fn = os.path.join('cache', fn)
    if os.path.exists(fn):
        print(f'{fn} exists, using cached version')
        return pd.read_csv(fn)
    else:
        print(f'{fn} does not exist, creating file')
        df = fun(*args, **kwargs)
        df.to_csv(fn, index=False)
        return df


def get_companies():
    codigos = cache_data('codigos.csv', get_listed_codes)
    setores = cache_data('setores.csv', get_sectors)
    numshares = cache_data('num_shares.csv', get_num_shares)

    df = pd.merge(codigos, setores, on='NM_PREGAO', how='inner')
    df = df.merge(numshares, on='BTICKER')
    return df.reset_index(drop=True)


def get_cvm_zip(year, doc_type, accounts=None, companies=None, rmzero=True):
    #
    fn = f'{doc_type.lower()}_cia_aberta_{year}'
    print('Downloading ' + fn)
    url = 'http://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/'
    if doc_type.lower() != 'itr':
        url = url + 'DFP/'
    url = url + doc_type.upper() + '/DADOS/' + fn + '.zip'
    #
    filehandle, _ = ur.urlretrieve(url)
    with ZipFile(filehandle, 'r') as zf:
        flist = zf.namelist()
        flist = [f for f in flist if 'con' in f]
        if fn + '.csv' in flist:
            flist.remove(fn + '.csv')
        df = pd.concat([
            pd.read_csv(io.BytesIO(zf.read(fn)), delimiter=';',
                        encoding='latin1')
            for fn in flist
        ])
    #
    if companies is not None:
        df = df[df['CD_CVM'].isin(companies)]
    if accounts is not None:
        df = df[df['CD_CONTA'].isin(accounts)]
    if rmzero:
        df = df[df['VL_CONTA'] != 0]
    #
    df['VL_CONTA'] = df['VL_CONTA'] * 10 ** \
        np.where(df['ESCALA_MOEDA'] == 'UNIDADE', 1, 3)
    #
    cols = df.columns
    cols = cols[cols.isin(['DT_REFER', 'VERSAO', 'CD_CVM',
                           'DT_INI_EXERC', 'DT_FIM_EXERC', 'CD_CONTA',
                           'DS_CONTA', 'VL_CONTA', 'COLUNA_DF'])]
    return df[cols].reset_index(drop=True)


def get_cvm_all(years, doc_types=['dre', 'bpa', 'bpp'],
                accounts=None, companies=None):
    doc_types.append('itr')
    df = (
        pd.concat([
            get_cvm_zip(year, doc_type, accounts, companies)
            for doc_type in doc_types
            for year in years
        ], ignore_index=True)
        .sort_values(['CD_CVM', 'CD_CONTA', 'DT_FIM_EXERC', 'DT_REFER',
                      'VERSAO'])
        .drop_duplicates(['CD_CVM', 'CD_CONTA', 'DT_FIM_EXERC'], keep='last')
        .assign(VL_CONTA=lambda x: x['VL_CONTA'] / 1000000)
        .rename(columns={'VL_CONTA': 'VL_CONTA_YTD'})
        .reset_index(drop=True)
    )
    df['VL_CONTA'] = np.where(
        df['CD_CONTA'].str[:1].isin(['1', '2']),
        df['VL_CONTA_YTD'],
        df['VL_CONTA_YTD'] -
            (df.groupby(['CD_CVM', 'CD_CONTA', 'DT_INI_EXERC'])['VL_CONTA_YTD']
            .shift(fill_value=0))
    )
    return df


def get_pib():
    url = "https://sidra.ibge.gov.br/geratabela?format=us.csv&" + \
        "name=tabela6613.csv&terr=N&rank=-&query=" + \
        "t/6613/n1/all/v/all/p/all/c11255/90687,90691,90696,90707/" + \
        "d/v9319%202/l/t,v%2Bc11255,p"
    df = (
        pd.read_csv(url, skiprows=5, skipfooter=11,
            names=['DT_FIM_EXERC', 'PIB_AGRO', 'PIB_IND', 'PIB_SERV', 'PIB'])
        .assign(
            DT_FIM_EXERC=lambda x: pd.to_datetime({
                'year': x['DT_FIM_EXERC'].str[-4:],
                'month': x['DT_FIM_EXERC'].str[0].astype(int) * 3,
                'day': 1
            })
        )
        .set_index('DT_FIM_EXERC')
        .resample('Q').last()
        .reset_index()
    )
    return df
