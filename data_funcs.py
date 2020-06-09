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
    numshares = numshares[['Código', 'Qtde Ações Ordinárias', 'Qtde Ações Preferenciais']]
    numshares.columns = ['BTICKER', 'ON', 'PN']
    numshares = numshares.groupby('BTICKER').sum().reset_index()
    numshares = numshares.melt('BTICKER', var_name='TIPO', value_name='QT_ACOES')
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


def get_companies(index_name='IBRA'):
    ibra = cache_data(f'{index_name}.csv', get_index_composition, 'IBRA')
    codigos = cache_data('codigos.csv', get_listed_codes)
    setores = cache_data('setores.csv', get_sectors)
    numshares = cache_data('num_shares.csv', get_num_shares)

    df = pd.merge(codigos, ibra, on='NM_PREGAO', how='inner')
    df['BTICKER'] = df['TICKER'].str[:4]
    df = df.merge(setores.drop(columns=['NM_PREGAO']), on='BTICKER')
    df['TIPO'] = df['TIPO'].str[:2]
    df = df.merge(numshares, on=['BTICKER', 'TIPO'])
    return df.reset_index(drop=True)
