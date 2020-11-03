

## importing the dataset
import pandas as pd
data=pd.read_csv('E:\\assignment\\neural network\\50_Startups.csv')
data.head()
data.describe()
data.info()
## so we do have 50 rows and 5 columns and we need to build a Neural Network model to predict profit 
## firstly lets convert the object column into int format
data['State'].unique()
data['State'],_ = pd.factorize(data['State'])


## as our dataset contains different values in different columns, so we need to normalize the data set to get a better result
## normalizing dataset
data_new=(data-data.min())/(data.max()-data.min())
data_new.head()
data_new.describe()


##selecting the target variales and the predictors
x=data_new.iloc[:,:4]
y=data_new.iloc[:,4]

##training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


##building and Training our First neural network
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([Dense(20, activation='sigmoid', input_shape=(4,)),Dense(20, activation='sigmoid'),Dense(1, activation='sigmoid'),])
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


import numpy  as np
first_model = model
first_model.fit(np.array(x_train),np.array(y_train),epochs=10)
y_pred = first_model.predict(np.array(x_test))
y_pred = pd.Series([i[0] for i in y_pred])
rmse_value = np.sqrt(np.mean((y_pred-y_test)**2))

## visualisation
import matplotlib.pyplot as plt
plt.plot(y_pred,y_test,"bo")
np.corrcoef(y_pred,y_test)

## so the corelation is found to be .911
