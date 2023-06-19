import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
dataset= pd.read_csv(r'C:\Users\enghu\OneDrive\Desktop\utd\ML\1st project\combined cycle\CCPP\New folder (2)\dataset2.csv')
#data pre-processing 
dataset.drop_duplicates()
dataset.dropna()
def normalize(x):
    mu=np.mean(x,axis=0)
    sgma=np.std(x,axis=0)
    x_norm=(x-mu)/sgma
    return x_norm,mu,sgma
x = dataset.drop(["y"],axis=1).values 
y = dataset['y'].values
d1=x.shape[0]
ones=np.ones((d1,1))
x=np.concatenate((ones,x),axis=1)
# data spliting
# initial coefficients
b = np.array([0,0,0, 0, 0]) 

def hypothesis(x, b): 
 return np.dot(x, b)
def cost_function(x, y, b):
 m = len(y)
 J = 1/(2*m) * np.sum((hypothesis(x, b) - y)**2) 
 return J 
def gradient_descent(x, y, b, learning_rate, num_iterations):
 m = len(y) 
 J_history = np.zeros(num_iterations)
 for i in range(num_iterations): 
   h = hypothesis(x, b) 
   gradient = 1/m * np.dot(x.T, (h - y)) 
   b = b - learning_rate * gradient 
   J_history[i] = cost_function(x, y, b) 
 return b, J_history
 # example usage with data split 

 
learning_rate = 0.01 
num_iterations = 1000
 # normalize input attributes 
x_norm, mu, sgma = normalize(x)
 # split data into training and test sets np.random.seed(42) 
# for reproducibility
indices = np.random.permutation(len(x)) 
split = int(0.8 * len(x)) 
train_indices, test_indices = indices[:split], indices[split:]
x_train, x_test = x_norm[train_indices], x_norm[test_indices] 
y_train, y_test = y[train_indices], y[test_indices] 
b_final, J_history = gradient_descent(x_train, y_train, b, learning_rate, num_iterations)
print("Final coefficients:", b_final)
print("Final cost function value:", J_history[-1])
 # evaluate on test set 
# normalize test set using training set statistics 
x_test_norm = (x_test - mu) / sgma 
J_test = cost_function(x_test_norm, y_test, b_final) 
print("Test set cost function value:", J_test) 






