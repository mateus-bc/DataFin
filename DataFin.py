import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as Dash

from pandas_datareader import data as web
from datetime import date, datetime

class DataFin:
    today = date.today().strftime("%m/%d/%Y")
    
    def __init__(self,
                tickers,
                start_date: str = '01/01/2018',
                end_date: str = today) :    
        self.tickers = tickers
        self.start = start_date
        self.end = end_date

        self.price = self.ETL_multi(tickers = self.tickers, start_date = self.start, end_date = self.end, variable = 'price') 
        self.trend = self.calc_trend()
        self.delta= self.calc_delta()

    def calc_trend(self,days=30):
        base=self.price.rolling(days).mean()
        self.trend = base
        return base
    
    def calc_delta(self,days=1):
        base=self.price.diff(days)
        self.delta = base
        return base

    def plot(self,norm=False,start=None,trend=None):

        base=self.price
        if trend:
            base = self.trend    

        if norm: 
            base = self.norm(start=start)


        dash = Dash.Figure()
        for asset in self.tickers: 
            dash.add_trace(Dash.Scatter(x= base.index,y=base[asset],name=asset,
                                    mode='lines',opacity=0.6))

        dash.update_layout(title='Análise Multi',titlefont_size=28,
                            xaxis= dict(title='Tempo',titlefont_size=16,tickfont_size=14),
                            yaxis= dict(title='Preço ($)',titlefont_size=16,tickfont_size=14),
                            height=550      
                        )
        return dash

    def calc(self,  reference=None, start=None, end =None, risk_free=None):


        if start == None: start = self.price.index[0]
        if end == None: end = self.price.index[-1]
        if reference == None: reference = self.tickers[-1]

        # Vetores ativos especiais
        base = norm(self.price)
        
        v_reference = base[reference]
        if risk_free == None:
            risk_free = reference 
            v_reference = base[risk_free]
        

        # Calculation
        rentabilidade = ((base.loc[end]/base.loc[start])-1) # Yield
        desvio_padrao = np.std(base[start:end]) # DP
        
        # BETA
        base_beta = base.diff(1)[1:]
        cov = {}
        for asset in base_beta.columns:
            cov[asset] =+ np.cov(base_beta[asset],base_beta[reference])[0][1]
        cov = np.array(list(cov.values()))
        beta = np.var(base_beta)/cov # SHARPE
        

        # Output Dictionary
        print(f'Reference: {reference}')
        stats = dict(
            rentabilidade = round(rentabilidade*100,2) ,
            dp = round(desvio_padrao,2),
            alpha = round(((rentabilidade-rentabilidade[reference])/rentabilidade[reference])*100,2),
            beta = beta,
            sharpe= (rentabilidade-rentabilidade[risk_free])/desvio_padrao
            )

        return pd.DataFrame(stats)    

    def norm(self,start=None):
        if start == None: start= self.price.index[0]
        df_norm = (self.price[start:]/self.price.loc[start])*100
        return df_norm

    def ETL_multi(self, tickers, start_date, end_date, variable):
        base = pd.DataFrame(ETL_single(tickers[0], start_date, end_date)[variable])
        base.rename({variable: tickers[0]},inplace=True,axis=1)
        
        for cod in tickers:
            base[cod]= ETL_single(cod, start_date, end_date)[variable] 
        return base

    def ETL_single(self, ticker, start_date, end_date):
        base = web.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)
        
        base = base[['Close','Volume','Adj Close']]
        base.rename({'Close':'price','Volume':'volume','Adj Close':'price adj'},axis=1,inplace=True)

        base = base[['price','price adj','volume']]
        return base




#=============== ETL ===============#

def ETL_setores():
    # Importar de base de empresas e setores B3

    cod = pd.read_csv('base_papeis.csv',sep=';') 

    # Base de Setores
    setores = cod.pivot_table(index=['setor','subsetor'],values='cod',aggfunc='count')
    setores.reset_index(inplace=True)
    setores.rename({'cod':'qtd'},axis=1,inplace=True)
    return (cod,setores)

def ETL_single(ticker,start_date,end_date,adjust_ticker=False,ticker_type=4):
    if adjust_ticker: ticker= f'{ticker}{ticker_type}.SA' #Arruma o nome do ticker para padrão do Yahoo Finance para B3    
    base = web.DataReader(ticker,data_source='yahoo',start=start_date,end=end_date)
    base = base[['Close','Volume','Adj Close']]
    base.rename({'Close':'price','Volume':'volume','Adj Close':'price adj'},axis=1,inplace=True)

    base['mean5'] = base['price'].rolling(5).mean() 
    base['mean30'] = base['price'].rolling(30).mean()
    base['diff'] = base['price'].diff(1)

    base = base[['price','mean5','mean30','diff','price adj','volume']]
    return base

def ETL_multi(tickers,start_date,end_date,variable):

    base=pd.DataFrame(ETL_single(tickers[0],start_date,end_date,adjust_ticker=False)[variable])
    base.rename({variable:tickers[0]},inplace=True,axis=1)
    
    for cod in tickers:
        base[cod]=ETL_single(cod,start_date,end_date,adjust_ticker=False)[variable] 

    return base

#=============== Analysis ===============#


def norm(df,start=None):
    if start == None: 
        start = df.index[0]
    else: 
        start = datetime.strptime(start, '%m/%d/%Y')
    
    df_norm = (df[start:]/df.loc[start])*100
    return df_norm
    
def calc(df,start,end,reference,risk_free=None):

    # Special Assets Vectors
    base = norm(df,start)
    v_reference = base[reference]
    if risk_free == None:
        risk_free = reference 
        v_reference = base[risk_free]
        
    # Time interval                 
    #start = datetime.strptime(start, '%m/%d/%Y')
    #end = datetime.strptime(end, '%m/%d/%Y')
    
    # Calculation
    rentabilidade = ((base.loc[end]/base.loc[start])-1) # Yield
    desvio_padrao = np.std(base) # DP
    
    # BETA
    base_beta = base.diff(1)[1:]
    cov = {}
    for asset in base_beta.columns:
        cov[asset] =+ np.cov(base_beta[asset],base_beta[reference])[0][1]
    cov = np.array(list(cov.values()))
    beta = np.var(base_beta)/cov # SHARPE
    
    # Output Dictionary
    stats = dict(
        rentabilidade = round(rentabilidade*100,2) ,
        dp = round(desvio_padrao,2),
        alpha = round(((rentabilidade-rentabilidade[reference])/rentabilidade[reference])*100,2),
        beta = beta,
        sharpe= (rentabilidade-rentabilidade[risk_free])/desvio_padrao
        )

    return pd.DataFrame(stats)

#=============== Plotting ===============#

def plot_single(base):
    # Plota gráficos de preço, média móvel 5 dias e média móvel 30 dias
    
    dash = Dash.Figure()
    dash.add_trace(Dash.Scatter(x= base.index,y=base.price,name='Preço',
                                 mode='lines',marker_color='#220cb0',opacity=0.6))
    dash.add_trace(Dash.Scatter(x= base.index,y=base.mean5,name='Média Móvel 5d',
                                 mode='lines',marker_color='#0c97b0',opacity=0.6))
    dash.add_trace(Dash.Scatter(x= base.index ,y=base.mean30,name='Média Móvel 30d',
                                 mode='lines',marker_color='#0cb09d',opacity=0.6))
    dash.update_layout(title='1. Análise Single',titlefont_size=28,
                        xaxis= dict(title='Tempo',titlefont_size=16,tickfont_size=14),
                        yaxis= dict(title='Preço ($)',titlefont_size=16,tickfont_size=14),
                        height=650      
                       )
    return dash

def plot_multi(base):
    dash = Dash.Figure()

    ativos = base.columns
    for ativo in ativos: 
        dash.add_trace(Dash.Scatter(x= base.index,y=base[ativo],name=ativo,
                                 mode='lines',opacity=0.6))

    dash.update_layout(title='2. Análise Multi',titlefont_size=28,
                        xaxis= dict(title='Tempo',titlefont_size=16,tickfont_size=14),
                        yaxis= dict(title='Preço ($)',titlefont_size=16,tickfont_size=14),
                        height=650      
                       )
    return dash

def plot_corr(base, ativos):
    base=base[ativos]
    dash = Dash.Figure()

    dash.add_trace(Dash.Scatter(x= base[ativos[0]],y=base[ativos[1]],
                                     mode='markers',marker_color='#220cb0',opacity=0.6))

    dash.update_layout(title='3. Scatter Plot ',titlefont_size=28,
                            xaxis= dict(title=ativos[0],titlefont_size=16,tickfont_size=14),
                            yaxis= dict(title=ativos[1],titlefont_size=16,tickfont_size=14),
                            height=650      
                           )
    return dash