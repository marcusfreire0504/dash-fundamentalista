import os
import io
import locale
import datetime
import numpy as np
import pandas as pd
import requests
import urllib.parse
import urllib.request as ur
from bs4 import BeautifulSoup
from zipfile import ZipFile
import xml.etree.ElementTree as ET




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


def get_quotes(tickers):
    url = 'http://bvmf.bmfbovespa.com.br/cotacoes2000/' + \
        'FormConsultaCotacoes.asp?strListaCodigos=' + '|'.join(tickers)
    page = requests.get(url)
    xml = ET.fromstring(page.text)
    df = pd.DataFrame([p.attrib for p in xml.findall('Papel')])
    df = df[['Codigo', 'Data', 'Ultimo']]
    df.columns = ['ticker', 'data', 'cotacao']
    df['cotacao'] = pd.to_numeric(df['cotacao'].str.replace(',','.'))
    return df


def get_mktcap():
    url = "http://www.b3.com.br/pt_br/market-data-e-indices/" + \
        "servicos-de-dados/market-data/consultas/mercado-a-vista/" + \
            "valor-de-mercado-das-empresas-listadas/bolsa-de-valores/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    url = soup.find("a", string="Histórico diário").get('href')
    url = "http://www.b3.com.br/" + url.replace('../', '')
    df = (
        pd.read_excel(url, skiprows=7, skipfooter=5)
        .dropna(axis=1, how="all")
        .rename(columns={"Empresa": "NM_PREGAO", "R$ (Mil)": "MarketCap"})
        .assign(MarketCap=lambda x: x['MarketCap'] / 1000)
        [["NM_PREGAO", "MarketCap"]]
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


def get_ipca():
    locale.setlocale(locale.LC_ALL,'pt_BR.UTF-8')
    url = "https://sidra.ibge.gov.br/geratabela?format=us.csv&" + \
        "name=tabela1737.csv&terr=N&rank=-&query=" + \
        "t/1737/n1/all/v/2266/p/all/d/v2266%2013/l/t%2Bv,,p"
    df = (
        pd.read_csv(url, skiprows=4, skipfooter=13,
                    names=['DT_FIM_EXERC', 'IPCA'])
        .assign(
            IPCA=lambda x: x['IPCA'] / x['IPCA'].iloc[-1],
            DT_FIM_EXERC=lambda x: pd.to_datetime(
                x['DT_FIM_EXERC'],
                format="%B %Y"
            )
        )
        .set_index('DT_FIM_EXERC')
        .resample('Q').last()
        .reset_index()
    )
    return df


def bcb_sgs(beg_date, end_date, **kwargs):
    return pd.concat([
        pd.read_json(f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{v}" +
                     f"/dados?formato=json&dataInicial={beg_date}&" +
                     f"dataFinal={end_date}",
                     convert_dates=False)
        .assign(DT_FIM_EXERC=lambda x: pd.to_datetime(x.data, dayfirst=True))
        .set_index('DT_FIM_EXERC')
        .rename(columns={'valor': k}) for k, v in kwargs.items()
    ], axis=1)


def get_usd():
    df = (
        bcb_sgs('01/01/1996', '31/03/2020', USD=3697)
        .resample('Q')
        ['USD']
        .agg(['last', 'mean'])
        .reset_index()
        .rename(columns={
            'last': 'USD_EOP',
            'mean': 'USD_AVG'
        })
    )
    return df


def last_friday():
    current_time = datetime.datetime.now()
    return str(
        current_time.date()
        - datetime.timedelta(days=current_time.weekday())
        + datetime.timedelta(days=4, weeks=-1))


def get_focus_quarterly(column='PIB Total', date=None):
    if date is None:
        date = last_friday()
    url = \
        "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/" + \
        "odata/ExpectativasMercadoTrimestrais?$top=100&$format=text/csv" + \
        "&$filter=Indicador%20eq%20'" + urllib.parse.quote(column) + \
        "'%20and%20Data%20eq%20'" + date + "'"
    dfq = (
        pd.read_csv(url, decimal=',')
        [['DataReferencia', 'Media', 'Mediana', 'Minimo', 'Maximo']]
        .assign(
            DataReferencia=lambda x: pd.to_datetime({
                'year': x['DataReferencia'].str[-4:],
                'month': x['DataReferencia'].str[0].astype(int) * 3,
                'day': 1
            })
        )
        .set_index("DataReferencia")
        .resample('Q').last()
    )
    return dfq


def get_focus_monthly(column="IPCA", date=None):
    if date is None:
        date = last_friday()
    url = \
        "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/" + \
            "odata/ExpectativaMercadoMensais?$top=100&$format=text/csv" + \
            "&$filter=(Indicador%20eq%20'" + urllib.parse.quote(column) + \
            "')%20and%20" + \
            "Data%20eq%20'" + date + "'%20and%20baseCalculo%20eq%200"
    dfm = (
        pd.read_csv(url, decimal=',')
        [['DataReferencia', 'Media', 'Mediana', 'Minimo', 'Maximo']]
        .assign(
            DataReferencia=lambda x: pd.to_datetime({
                'year': x['DataReferencia'].str[-4:],
                'month': x['DataReferencia'].str[:2],
                'day': 1
            })
        )
        .set_index('DataReferencia')
        .resample('M').last()
    )
    return dfm


def get_focus_yearly(column="IPCA", date=None):
    if date is None:
        date = last_friday()
    url = \
        "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/" + \
        "odata/ExpectativasMercadoAnuais?$top=100&$format=text/csv" + \
        "&$filter=Indicador%20eq%20'" + urllib.parse.quote(column) + \
        "'%20and%20Data%20eq%20'" + date + "'"
    dfy = (
        pd.read_csv(url, decimal=",")
        [['DataReferencia', 'Media', 'Mediana', 'Minimo',
          'Maximo']]
        .assign(
            DataReferencia=lambda x: pd.to_datetime({
                'year': x['DataReferencia'],
                'month': 12,
                'day': 1
            })
        )
        .set_index('DataReferencia')
        .resample('Y').last()
    )
    return dfy


def get_focus(historicals, date=None):
    if date is None:
        date = last_friday()

    pd.set_option('display.max_rows', None)
    date = last_friday()
    pib_q = get_focus_quarterly('PIB Total', date)
    pib_y = get_focus_yearly('PIB Total', date)
    ipca_m = get_focus_monthly('IPCA', date)
    ipca_y = get_focus_yearly('IPCA', date)
    usd_m = get_focus_monthly('Taxa de câmbio', date)
    usd_y = get_focus_yearly('Taxa de câmbio', date)

    ipca = (
        pd.concat([
            ipca_m/100,
            (1 + ipca_y[ipca_y.index.isin(ipca_m.index) == False]/100)**(1/12)-1
        ])
        .resample('M').last()
        .fillna(method="backfill")
    )
    ipca = (1 + ipca).resample('Q').prod() - 1

    pib = (
        pd.concat([
            pib_q/100, pib_y[pib_y.index.isin(pib_q.index) == False]/100
        ])
        .resample('Q').last()
        .fillna(method="backfill")
    ) * 100

    usd = (
        pd.concat([
            usd_m,
            usd_y[usd_y.index.isin(usd_m.index) == False]
        ])
        .resample("M").last()
        .interpolate()
        .resample("Q").mean()
    )

    macro = historicals.tail(4)

    ipca_idx = macro['IPCA'].iloc[3] * np.cumprod(1+ipca)

    pib2 = pib.copy()
    pib2.iloc[:,:] = np.NaN
    pib2.iloc[:4,:] = (1 + pib.iloc[:4,:]) *  macro[['PIB']].values
    while pib2.isna().sum().sum() > 0:
        pib2.iloc[:,:] = np.where(pib2.isna(), pib2.shift(4) * (1 + pib), pib2)

    df = (
        pib.reset_index().melt("DataReferencia", value_name="PIB")
        .merge(
            pib2.reset_index().melt("DataReferencia", value_name="PIBIndex"),
            how="outer"
        )
        .merge(
            ipca.reset_index().melt("DataReferencia", value_name="IPCAVar"),
            how="outer"
        )
        .merge(
            ipca_idx.reset_index().melt("DataReferencia", value_name="IPCA"),
            how="outer"
        )
        .merge(
            usd.reset_index().melt("DataReferencia", value_name="USD"),
            how="outer"
        )
        .rename(columns={'variable': 'scenario', 'DataReferencia': 'DT_FIM_EXERC'})
    )
    return df