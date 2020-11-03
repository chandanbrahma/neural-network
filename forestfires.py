## importing the dataset
import pandas as pd

data=pd.read_csv('E:\\assignment\\neural network\\forestfires.csv')

data.head()
data.describe()
data.info()
## so we do have 517 rows and 31 columns and we need to build a Neural Network model to predict profit 


## converting the catagorical variable into integer format
data.month,_ = pd.factorize(data.month)

data.day,_ = pd.factorize(data.day)

data['size_category'],_ = pd.factorize(data['size_category'])


data.head()
data.info()

##normalizing the data
data_new=(data-data.min())/(data.max()-data.min())
data_new.head()
data_new.describe() 

##building model
##selecting the prectors and the target variables
x= data_new.iloc[:,:-1]
y= data_new.iloc[:,-1]

##training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


##building and Training our First neural network
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30,input_dim=30,activation="relu"))###8 input
model.add(Dense(20,activation="relu"))##hidden layer2
model.add(Dense(10,activation="relu"))##hidden layer3
model.add(Dense(1,kernel_initializer="normal"))### output
model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])


model = Sequential([Dense(50, activation='relu', input_shape=(30,)),Dense(20, activation='relu'),Dense(1, activation='relu'),])
model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"]) ##calculating the errors so accordingly weights will be assigned


import numpy  as np
first_model = model
first_model.fit(np.array(x_train),np.array(y_train),epochs=10)
y_pred = first_model.predict(np.array(x_test))
y_pred = pd.Series([i[0] for i in y_pred])
rmse_value = np.sqrt(np.mean((y_pred-y_test)**2))

import matplotlib.pyplot as plt
plt.plot(y_pred,y_test,"bo")
np.corrcoef(y_pred,y_test)
