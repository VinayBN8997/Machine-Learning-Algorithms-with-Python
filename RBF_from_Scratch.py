
import numpy as np
import pandas as pd
import random
from sklearn import model_selection
from sklearn.cluster import KMeans 
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import norm, pinv


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.radius = np.zeros(numCenters)
        self.kNeighbors = 3
        self.W = np.random.random((self.numCenters, self.outdim))
        
    def basisfunc(self, c, d , ci):
        return np.exp(-(1 / self.radius[ci]) * norm(c-d)**2)
     
    def calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self.basisfunc(c, x,ci)
        return G
     
    def fit(self, X, Y):
        kMeans = KMeans(n_clusters = self.numCenters)
        kMeans = kMeans.fit(X)
        self.centers = kMeans.cluster_centers_
        kNearest = NearestNeighbors(n_neighbors=self.kNeighbors, algorithm='ball_tree').fit(self.centers)
        distances, indices = kNearest.kneighbors(self.centers)
        for i in range(self.numCenters):
            self.radius[i] = np.sqrt((norm(distances[i])**2 )/ self.kNeighbors)
        # calculate activations of RBFs
        G = self.calcAct(X)
        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)
         
    def predict(self, X):
        G = self.calcAct(X)
        Y = np.dot(G, self.W)
        for i in range(len(Y)):
            if Y[i] > 0.5:
                Y[i] = 1
            else:
                Y[i] = 0
        return Y

df= pd.read_csv("Heart_data_labels.csv")
for i in [2,3,4]:
    df['Out'].replace(i,1,inplace = True)
df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Trestbps"] = (df["Trestbps"] - df["Trestbps"].min()) / (df["Trestbps"].max() - df["Trestbps"].min())
df["Chol"] = (df["Chol"] - df["Chol"].min()) / (df["Chol"].max() - df["Chol"].min())
df["thalach"] = (df["thalach"] - df["thalach"].min()) / (df["thalach"].max() - df["thalach"].min())
X = np.array(df.drop("Out",axis = 1))
y = np.transpose(np.array([df["Out"]]))
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.3)
data = np.array(df)
    
indim = X_train.shape[1]
num_centers = 20
outdim = 1
rbf = RBF(indim, num_centers, outdim)
rbf.fit(X_train, y_train)

prediction = rbf.predict(X_test)
correct_count = 0.0
a = 0.0
b = 0.0
c = 0.0
d = 0.0
for i in range(len(X_test)): 
    if y_test[i] == prediction[i]:
        correct_count += 1
        if y_test[i] ==1:
            a+=1
        else:
            d+=1
    else:
        if y_test[i] ==1:
            b+=1
        else:
            c+=1

p=a/(a+c)
r=a/(a+b)
print("Accuracy: " + str((correct_count/len(X_test))*100))
print("Precision: " + str((a/(a+c))))
print("Recall: " + str((a/(a+b))))
print("F measure: " + str(2*p*r/(p+r)))
