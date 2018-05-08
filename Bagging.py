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
bagging_no = 10

#KNN
from sklearn import neighbors

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
for k in [1,3,5,9,15]:
    clf = list()
    prediction = list()
    bagged_prediction = list()
    for i in range(bagging_no):
        idx[i] = np.random.randint(len(Train_data), size=150)
        Sub_Train_data = Train_data[np.array(idx[i]),:]
        X_train = Sub_Train_data[:,:-1]
        y_train = Sub_Train_data[:,-1]
        clf.append(neighbors.KNeighborsClassifier(n_neighbors = k))
        clf[i].fit(X_train,y_train)
    for i in range(bagging_no):
        prediction.append(clf[i].predict(X_test))
    prediction = np.array(prediction)
    for i in range(len(X_test)):
        bagged_prediction.append(mode(prediction[:,i])[0][0])
    correct_count = 0.0
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    for i in range(len(X_test)): 
        if y_test[i] == bagged_prediction[i]:
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

    print("KNN K:",k,"Accuracy: ",round(Accuracy,2),"%")


#DT

from sklearn.tree import DecisionTreeClassifier

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
bagged_prediction = list()
for i in range(bagging_no):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(DecisionTreeClassifier())
    clf[i].fit(X_train,y_train)
for i in range(bagging_no):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    bagged_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == bagged_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("DT Accuracy: %.2f " % Accuracy)

#RBF

from RBF import RBF

indim = X_train.shape[1]
num_centers = 20
outdim = 1

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
bagged_prediction = list()
for i in range(bagging_no):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(RBF(indim, num_centers, outdim))
    clf[i].fit(X_train,y_train)
for i in range(bagging_no):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    bagged_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0
for i in range(len(X_test)): 
    if y_test[i] == bagged_prediction[i]:
        correct_count += 1
Accuracy = ( correct_count/len(X_test) ) * 100
print("RBF Accuracy: ",round(Accuracy,2),"%")


#SVM
from sklearn import svm

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
bagged_prediction = list()
for i in range(bagging_no):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(svm.SVC(kernel = "linear"))
    clf[i].fit(X_train,y_train)
for i in range(bagging_no):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    bagged_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0.0
a = 0.0
b = 0.0
c = 0.0
d = 0.0
for i in range(len(X_test)): 
    if y_test[i] == bagged_prediction[i]:
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

#MLP
from sklearn.neural_network import MLPClassifier

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
bagged_prediction = list()
for i in range(bagging_no):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(MLPClassifier(hidden_layer_sizes=(12,7,6),max_iter=2000))
    clf[i].fit(X_train,y_train)
for i in range(bagging_no):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    bagged_prediction.append(mode(prediction[:,i])[0][0])

correct_count = 0.0
a = 0.0
b = 0.0
c = 0.0
d = 0.0
for i in range(len(X_test)): 
    if y_test[i] == bagged_prediction[i]:
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

#NB
from sklearn.naive_bayes import GaussianNB

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
Train_data = np.append(X_train ,y_train , axis = 1)
idx = np.zeros((10,150), dtype=int)
clf = list()
prediction = list()
bagged_prediction = list()
for i in range(bagging_no):
    idx[i] = np.random.randint(len(Train_data), size=150)
    Sub_Train_data = Train_data[np.array(idx[i]),:]
    X_train = Sub_Train_data[:,:-1]
    y_train = Sub_Train_data[:,-1]
    clf.append(GaussianNB())
    clf[i].fit(X_train,y_train)
for i in range(bagging_no):
    prediction.append(clf[i].predict(X_test))
prediction = np.array(prediction)
for i in range(len(X_test)):
    bagged_prediction.append(mode(prediction[:,i])[0][0])
correct_count = 0.0
a = 0.0
b = 0.0
c = 0.0
d = 0.0
for i in range(len(X_test)): 
    if y_test[i] == bagged_prediction[i]:
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
