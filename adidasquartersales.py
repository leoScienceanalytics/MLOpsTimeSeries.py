#Bibliotecas
import pandas as pd
from datetime import date, timedelta
import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings
import plotly.graph_objects as go
import plotly.express as px


#Importando dataset
data = pd.read_csv('adidasquarterlysales.csv')
print(data)


#Data Cleanning
data['Time Period'] = pd.to_datetime(data['Time Period'])


#Describe Stats
print(data.describe())



#Tendency, Seasonal and Residual
result = seasonal_decompose(data["Revenue"], 
                            model='multiplicative', period = 30) 
fig = result.plot()  
fig.set_size_inches(15, 10)
plt.show()


# Autocorrelation Graphic
pd.plotting.autocorrelation_plot(data["Revenue"]) #Para achar o P
# pd.plot é um módulo de funções e autocorrelation_plot é uma das funções que irá clacular e plotar a autocorrelação
plt.title('Autocorrelation Graphic, defines P')
plt.show()

#Parcial Autocorrelation
plot_pacf(data["Revenue"], lags = 20) #Para achar o Q
plt.title('Parcial Autocorrelation Graphic, defines Q')
plt.show()



model=sm.tsa.statespace.SARIMAX(data['Revenue'],
                                order=(5, 1, 2),
                                seasonal_order=(5, 1, 2, 12))
model=model.fit()
print(model.summary())

predictions = model.predict(len(data), len(data)+7)
print(predictions)

data["Revenue"].plot(legend=True, 
                     label="Training Data", 
                     figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")
plt.show()

