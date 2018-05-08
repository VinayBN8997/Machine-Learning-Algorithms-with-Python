
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


