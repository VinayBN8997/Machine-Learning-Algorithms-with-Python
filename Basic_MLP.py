import numpy as np
import pandas as pd
import random

df= pd.read_csv("Heart_data_labels.csv")
for i in [2,3,4]:
    df['Out'].replace(i,1,inplace = True)
df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Trestbps"] = (df["Trestbps"] - df["Trestbps"].min()) / (df["Trestbps"].max() - df["Trestbps"].min())
df["Chol"] = (df["Chol"] - df["Chol"].min()) / (df["Chol"].max() - df["Chol"].min())
df["thalach"] = (df["thalach"] - df["thalach"].min()) / (df["thalach"].max() - df["thalach"].min())
df.insert(14,"Bias",1)
X = np.array(df.drop("Out",axis = 1))
y = np.transpose(np.array([df["Out"]]))
neta_o = 0.1


def non_linear(X,derivative = False):#sigmoidal activation function
    if derivative == True:
        return X*(1-X)
    else:
        return 1/(1+np.exp(-X))


def non_linear_relu(X,derivative = False):#relu activation function
    if derivative == True:
        X[X<=0] = 0
        X[X>0] = 1
        return X
    else:
        X[X<0]=0
        return X

#Initialising the weights
w = [0,0,0,0]
w[0] = np.random.random((X.shape[1],12))
w[1] = np.random.random((12,8))
w[2] = np.random.random((8,6))
w[3] = np.random.random((6,1))



for j in range(90000):
    np.random.shuffle(X)
    neta = neta_o/(1 + (j/10000))
    #Forward Propogation
    l0 = X
    l1 = non_linear(np.dot(l0,w[0]))
    l2 = non_linear(np.dot(l1,w[1]))
    l3 = non_linear(np.dot(l2,w[2]))
    l4 = non_linear(np.dot(l3,w[3]))
    l4_error = y - l4  
    if (j % 10000) == 0:
        print("Error : " ,(1/len(X))*(np.sum(np.square(l4_error)))) #Error calculation for every 10000th iteration
    #Back Propogation
    l4_delta = l4_error * non_linear(l4,derivative = True)
    l3_error = l4_delta.dot(w[3].T) 
    l3_delta = l3_error * non_linear(l3,derivative = True)
    l2_error = l3_delta.dot(w[2].T)
    l2_delta = l2_error * non_linear(l2,derivative = True) 
    l1_error = l2_delta.dot(w[1].T) 
    l1_delta = l1_error * non_linear(l1,derivative = True) 
    #Updation of weights
    w[3] += neta*l3.T.dot(l4_delta)
    w[2] += neta*l2.T.dot(l3_delta)
    w[1] += neta*l1.T.dot(l2_delta)
    w[0] += neta*l0.T.dot(l1_delta)
    
#These Weights can be used for Predicting the test set of data
