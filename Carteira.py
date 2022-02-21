from matplotlib.cbook import simple_linear_interpolation
import numpy as np
import pandas as pd

from pandas_datareader import data as web
from datetime import date, datetime
from DataFin import DataFin, ETL_multi, ETL_single

start = '01/01/2018'
end = date.today().strftime("%m/%d/%Y")

class Carteira:
    
    today = date.today().strftime("%m/%d/%Y")
    input_path = 'C:/Users/mband/Desktop/1_FIN/2. Invest/Carteira.xlsx'
    
    def __init__(self,
                start_date: str = '01/01/2018',
                end_date: str = today,
                input_path: str = input_path):

        self.start = start_date   
        self.end = end_date   
        self.input_path: str = input_path

        self.df_carteira: pd.DataFrame = self.read('Carteira') 
        self.df_indices: pd.DataFrame = self.read('Índices') 

        self.carteira: pd.DataFrame = self.ETL_carteira()
        
        self.indices_tickers = list(self.df_indices['ticker'])
        self.assets_tickers = list(self.carteira['ticker'])
        self.tickers = self.assets_tickers + self.indices_tickers 
        
        self.part = self.carteira['pos']
        self.buy_price = self.carteira['buy_price']
        
        self.price: pd.DataFrame = self.ETL_multi(self.tickers,start_date=start,end_date=end,variable='price')
        self.indices: pd.DataFrame = self.ETL_multi(self.indices_tickers,start_date=start,end_date=end,variable='price')
        

    def ETL_carteira(self):
        carteira = self.df_carteira[['ticker','pos','buy_price','buy_date','moeda','kind','corr']]
        carteira = carteira[carteira['ticker']!='Vinci Valorem']
        carteira['pos'] = carteira['pos'].astype(float)
        return carteira 

    def ETL_multi(self, tickers, start_date, end_date, variable):
        base = pd.DataFrame(self.ETL_single(tickers[0], start_date, end_date)[variable])
        base.rename({variable: tickers[0]},inplace=True,axis=1)
        
        for cod in tickers:
            base[cod]= self.ETL_single(cod, start_date, end_date)[variable] 
        return base

    def ETL_single(self, ticker, start_date, end_date):
        base = web.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)
        
        base = base[['Close','Volume','Adj Close']]
        base.rename({'Close':'price','Volume':'volume','Adj Close':'price adj'},axis=1,inplace=True)

        base = base[['price','price adj','volume']]
        return base
    
    def vol(self):
        pd.DataFrame({ 'Crypto':self.df_carteira.calc(reference='^BVSP').iloc[10:12].mean()['dp'],
              'Ações BR':self.df_carteira.calc(reference='^BVSP').iloc[:8].mean()['dp'],
            'ETF':self.df_carteira.calc(reference='^BVSP').iloc[8:10].mean()['dp']},index=['Desvio Padrão']).T

    def DataFin(self):
        return DataFin(self.tickers,start_date=self.start,end_date=self.end)

    def read(self,sheet_name):
        return pd.read_excel(self.input_path,sheet_name=sheet_name) 
    
    def export(self):
        self.price.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_price.csv',sep = ';',decimal=',')
        self.indices.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_indices.csv',sep = ';',decimal=',')

   




def ETL_BI(start='01/01/2018',end=None):
    if end == None:     
        end = date.today().strftime("%m/%d/%Y") 

    carteira = pd.read_excel('C:/Users/mband/Desktop/1_FIN/2. Invest/Carteira.xlsx',sheet_name='Carteira')
    indices = pd.read_excel('C:/Users/mband/Desktop/1_FIN/2. Invest/Carteira.xlsx',sheet_name='Índices')

    # ETL Cateira
    carteira = carteira[['ticker','pos','buy_price','buy_date','moeda','kind','corr']]
    carteira = carteira[carteira['ticker']!='Vinci Valorem']
    carteira['pos'] = carteira['pos'].astype(float) 

    # Definição dos tickers da carteira e importação da base de preços
    tickers = list(carteira['ticker'])
    part = carteira['pos']
    buy_price = carteira['buy_price']

    # Bases de output 
    base_indices = ETL_multi(list(indices['ticker']),start_date=start,end_date=end,variable='price')
    base_price = ETL_multi(tickers,start_date=start,end_date=end,variable='price')

    base_portfolio = base_price*part
    base_portfolio['Carteira'] = base_portfolio.sum(axis=1)

    base_rentabilidade = base_price/buy_price-1

    base_pesos = base_portfolio[base_portfolio.columns[:-1]].div(base_portfolio[base_portfolio.columns[-1]],axis=0)

    # Exportação
    base_price.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_price.csv',sep = ';',decimal=',')
    base_portfolio.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_portfolio.csv',sep = ';',decimal=',')
#   base_rentabilidade.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_rentabilidade.csv',sep = ';',decimal=',')
    base_pesos.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_pesos.csv',sep = ';',decimal=',')
    base_indices.to_csv('C:/Users/mband/Desktop/1_FIN/2. Invest/Bases_BI/base_indices.csv',sep = ';',decimal=',')