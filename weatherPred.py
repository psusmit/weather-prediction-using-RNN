# =============================================================================
# CODED BY susmit410
# =============================================================================

# Importing the libraries(for RNN)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('weather_train.csv')
dataset=dataset.iloc[:,5:6].values
#feature scaling(scale range 0-1)
sc=MinMaxScaler()
dataset=sc.fit_transform(dataset)
#get input and outputs
#x=input lenge calc with y-var we get output!!
x_train=dataset[0:311]
y_train=dataset[1:312]
#reshaping(only the input parameter)
x_train=np.reshape(x_train,(311,1,1))

#initialise the RNN!!
regressor=Sequential()
#add input layer and LSTM Layer(extra input layer)(by default 4 units)
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
#adding output layer
regressor.add(Dense(units=1))
#compile the model(b'coz it is regression not classification(mean_squared_error))
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
#fitting the model
regressor.fit(x_train,y_train,batch_size=32,epochs=200)

#getting the real stock prices
test_set = pd.read_csv('weather_test.csv')
real_temp=test_set.iloc[:,5:6].values
#get predicted stock prices
inputs=real_temp
inputs=sc.fit_transform(inputs)
inputs=np.reshape(inputs,(48,1,1))
#pred stock prices
predicted_temp=regressor.predict(inputs)
predicted_temp=sc.inverse_transform(predicted_temp)

#visualizing the results
plt.plot(real_temp,color='red',label='real')
plt.plot(predicted_temp,color='blue',label='predicted')
plt.title('mumbai temperature prediction')
plt.xlabel('Hours')
plt.ylabel('temperature in degree celcius')
plt.legend()
plt.show()

#evaluating the RNN
rmse=math.sqrt(mean_squared_error(real_temp,predicted_temp))
























