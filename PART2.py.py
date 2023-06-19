import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# loading data
dataset= pd.read_csv(r'C:\Users\enghu\OneDrive\Desktop\utd\ML\1st project\combined cycle\CCPP\New folder (2)\dataset2.csv')
#data pre-processing 
dataset.drop_duplicates()
dataset.dropna()
x = dataset.drop(["y"],axis=1).values 
y = dataset['y'].values
#data spliting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
from sklearn.linear_model import LinearRegression

LR=LinearRegression()
#train the model
LR.fit(x_train,y_train)
#predict for the test data
y_pred=LR.predict(x_test)
y_pred_train =LR.predict(x_train)

mse=mean_squared_error(y_train,y_pred_train)

print(y_pred)




