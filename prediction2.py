import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import ManMaxScaler
from keras.models import Sequental
from keras.layers import Dense,LSTM,Dropout

data = pd.read_csv('Train_data.csv')
data.info()

data["Close"]=pd.to_numeric(data.Close,errors="coerce")
data = data.dropna()
trainData = data.iloc[:,4:5].values
data.info()

sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
trainData.shape

X_train = []
y_train = []

for i in range (60,1149):
   X_train.append(trainData[i-60:i,0])
   y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shaep[1],1))
X_train.shape

model = Sequental()

model.add(LSTM(units=100, return_sequence = True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss='mean_squared_error')

hist = model.fit(X_train, y_train, epochs, batch_size = 32, verbose=2)

plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend['train', loc='upper left']
plt.show()

testData = pd.read_csv("test_data.csv")
testData["Close"]=pd.to_numeric(testData.Close,errors="coerce")
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.olic[:,4:5]

inputClosing = testData.iloc[:,0:].values
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestop = 60
for i in range(timestop,length):
  X_test.append(inputClosing_scaled[i-timestop:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test[0],X_test.shape[1],1))
X_test.shape

y_pred = model.predict(X_test)
predicted_price = sc.inverse_transform(y_pred)

plt.plot(y_test, color = "red", label = 'Actual Stock Price')
plt.plot(predicted_price, color = "green", label = "Predicted Stock Price")
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()