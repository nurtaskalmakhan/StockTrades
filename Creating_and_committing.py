import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)

tesla = pd.read_csv('TSLA.csv')

tesla.info()

tesla['Date'] = pd.to_datetime(tesla['Date'])

print(f'Dateframe contains stock prices between {tesla.Date.min()}{tesla.Date.max()}')
print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days} days')

tesla.describe()

tesla[['Open','High','Low','Close','Adj Close']].plot(kind='box')

layout = go.Layout(
    title = 'SPoT',
    xaxis = dict(
        title='Date',
        titlefont = dict(
            family='Courier New, monospace',
            size=18,
            color='##7f7f7f'
        )
    ),
    yaxis = dict(
        title='Price',
        titlefont = dict(
            family='Courier New, monospace',
            size=18,
            color='##7f7f7f'
        )
    )
)

tesla_data = [{'x':tesla['Date'], 'y':tesla['Close']}]
plot = go.figure(data=tesla_data, layout=layout)

iplot(plot)


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

X = np.array(tesla.index).reshape(-1,1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

Scaler = StandardScaler().fit(X_train)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)

trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)

tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot = go.Figure(data=tesla_data, layout=layout)

iplot(plot2)

scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{mse(Y_train, lm_predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)