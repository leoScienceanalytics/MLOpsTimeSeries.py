import pandas as pd #Biblioteca Pandas para estrutura de dados.
import yfinance as yf #API do Yahoo Finance; Preço do Bitcoin.
import datetime #Importa o tempo
from datetime import date, timedelta
import plotly.graph_objects as go #Gráfico na WEB
from autots import AutoTS #Biblioteca relevante para realizar previsões por séries temporais
import numpy as np #Biblioteca de cálculos matématicos
import matplotlib.pyplot as plt #Gráficos na máquina local
import time

def calcularpredict():
    today = date.today()


    d1 = today.strftime("%Y-%m-%d") #Formata o dia de Hoje
    end_date = d1 #Armazena o valor
    d2 = date.today() - timedelta(days=730) #Calcula a data inicial (730 dias antes)
    d2 = d2.strftime("%Y-%m-%d") #Formata essa data inicial
    start_date = d2 #armazena a data inicial


    #Importando dataset
    data = yf.download('BTC-USD', 
                        start=start_date, 
                        end=end_date, 
                        progress=False) #Conecta o dataset dos valores do bitcoin, com parâmetros as data calculadas

    #Criando DataFrame
    data["Date"] = data.index #Adiciona a coluna das datas ao DataFrame ['Date']
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]] #Reorganiza as colunas do DataFrame 'Data'
    data.reset_index(drop=True, inplace=True) #Redefine o index do DataFrame

    
    #Análise Exploratória
    print(data.head()) #Mostrar 5 primeiros dados

    print('Estatísticas Descritivas: ', data.describe())

    correlac = data.corr()
    print(correlac['Close'].sort_values(ascending=False))

    #Gráfico de séries temporiais
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                            open=data["Open"], 
                                            high=data["High"],
                                            low=data["Low"], 
                                            close=data["Close"])])

    figure.update_layout(title = "Bitcoin Price Analysis", 
                        xaxis_rangeslider_visible=False)

    figure.show() #Mostra cada coluna no período de tempo

    model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple') #Criar parâmetros do modelo
    #infer --> diz que o modelo vai aprender automaticamente a periodicidade dos dados
    #Simple --> diz que o conjunto do modelo é tido como simples
    model = model.fit(data, date_col='Date', value_col='Close', id_col=None) #Treinamento do modelo, seguindo os parâmetros estabelecidos anteriormente
    prediction = model.predict() #Modelo treinado é usado para prever
    forecast = prediction.forecast # valores previstos em prediction são armazenados em forecast
    forecast = pd.DataFrame(forecast)
    forecast = round(forecast, 2)
    print(forecast)
    
    
#Laço de repetição que permite que o gráfico seja atualizado a cada 10 Minutos 
uploadtime = 600    
try:
    while True:
        calcularpredict()
        time.sleep(uploadtime)
except KeyboardInterrupt:
    print('A atualização foi interrompida.')        
    











