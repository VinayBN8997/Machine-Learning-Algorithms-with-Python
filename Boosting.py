
import numpy as np
import pandas as pd
import random
from sklearn import model_selection
from scipy.stats import mode

df= pd.read_csv("Heart_data_labels.csv")
for i in [2,3,4]:
    df['Out'].replace(i,1,inplace = True)
df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Trestbps"] = (df["Trestbps"] - df["Trestbps"].min()) / (df["Trestbps"].max() - df["Trestbps"].min())
df["Chol"] = (df["Chol"] - df["Chol"].min()) / (df["Chol"].max() - df["Chol"].min())
df["thalach"] = (df["thalach"] - df["thalach"].min()) / (df["thalach"].max() - df["thalach"].min())
X = np.array(df.drop("Out",axis = 1))
y = np.transpose(np.array([df["Out"]]))
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.4)
data = np.array(df)
       
#DT
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)
bdt.fit(X_train, y_train)
boosted_prediction= bdt.predict(X_test)
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == boosted_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("DT Accuracy: ",round(Accuracy,2),"%")


#MLP cannot be Boosted using the library ( Sample Weight is not a parameter for this classifier)

'''#MLP
from sklearn.neural_network import MLPClassifier

bdt = AdaBoostClassifier(MLPClassifier(hidden_layer_sizes=(12,7,6),max_iter=2000),n_estimators=200)
bdt.fit(X_train, y_train)
boosted_prediction= bdt.predict(X_test)
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == boosted_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("MLP Accuracy: ",round(Accuracy,2),"%")
'''


#SVM
from sklearn import svm

bdt = AdaBoostClassifier(svm.SVC(kernel = "linear"),n_estimators=200,algorithm="SAMME")
bdt.fit(X_train, y_train)
boosted_prediction= bdt.predict(X_test)
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == boosted_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("SVM Accuracy: ",round(Accuracy,2),"%")

#NB
from sklearn.naive_bayes import GaussianNB

bdt = AdaBoostClassifier(GaussianNB(),n_estimators=200,algorithm="SAMME")
bdt.fit(X_train, y_train)
boosted_prediction= bdt.predict(X_test)
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == boosted_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("NB Accuracy: ",round(Accuracy,2),"%")


#RBF could not be boosted using this library ( Sample Weight is not a parameter for this classifier)

'''
#RBF
from RBF import RBF
indim = X_train.shape[1]
num_centers = 20
outdim = 1

bdt = AdaBoostClassifier(RBF(indim, num_centers, outdim),n_estimators=200,algorithm="SAMME")
bdt.fit(X_train, y_train)
boosted_prediction= bdt.predict(X_test)
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == boosted_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("RBF Accuracy: ",round(Accuracy,2),"%")
'''

#KNN
from sklearn import neighbors
for k in [1,3,5,9,15]:
    bdt = AdaBoostClassifier(neighbors.KNeighborsClassifier(n_neighbors = k),n_estimators=200,algorithm="SAMME")
    bdt.fit(X_train, y_train)
    boosted_prediction= bdt.predict(X_test)
    correct_count = 0
    for i in range(len(X_test)): 
        if y_test[i] == boosted_prediction[i]:
            correct_count += 1
    Accuracy = ( correct_count/len(X_test) ) * 100
    print("KNN K:",k,"Accuracy: ",round(Accuracy,2),"%")
