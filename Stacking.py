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
data = np.array(df)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)


from sklearn import svm
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from RBF import RBF
indim = X_train.shape[1]
num_centers = 20
outdim = 1
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

#SVM,RBF,MLP,KNN (9), NB

Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
stacked_prediction = list()

models=[svm.SVC(kernel = "linear"),RBF(indim, num_centers, outdim),MLPClassifier(hidden_layer_sizes=(12,7,6),max_iter=2000),neighbors.KNeighborsClassifier(n_neighbors = 9),GaussianNB()]
for i in range(len(models)):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(models[i])
    clf[i].fit(X_train,y_train)
for i in range(len(models)):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    stacked_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == stacked_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("Stacked [SVM,RBF,MLP,KNN (9), NB] Accuracy: %.2f " % Accuracy)

#SVM,RBF,MLP, NB
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
stacked_prediction = list()

models=[svm.SVC(kernel = "linear"),RBF(indim, num_centers, outdim),MLPClassifier(hidden_layer_sizes=(12,7,6),max_iter=2000),GaussianNB()]
for i in range(len(models)):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(models[i])
    clf[i].fit(X_train,y_train)
for i in range(len(models)):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    stacked_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == stacked_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("Stacked [SVM,RBF,MLP, NB] Accuracy: %.2f " % Accuracy)

#DT, MLP, SVM
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
stacked_prediction = list()

models=[svm.SVC(kernel = "linear"),MLPClassifier(hidden_layer_sizes=(12,7,6),max_iter=2000),DecisionTreeClassifier()]
for i in range(len(models)):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(models[i])
    clf[i].fit(X_train,y_train)
for i in range(len(models)):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    stacked_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == stacked_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("Stacked [DT,SVM,MLP] Accuracy: %.2f " % Accuracy)


#MLP,SVM
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
stacked_prediction = list()

models=[svm.SVC(kernel = "linear"),MLPClassifier(hidden_layer_sizes=(12,7,6),max_iter=2000)]
for i in range(len(models)):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(models[i])
    clf[i].fit(X_train,y_train)
for i in range(len(models)):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    stacked_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == stacked_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("Stacked [SVM,MLP] Accuracy: %.2f " % Accuracy)
