from math import sqrt
import random
import csv

def dummy(elem):
    return elem[0]

def KNearestNeighbours(Training,testCase,K):
    distance_table = []
    output_table = []
    for i in Training:
        distance = 0
        for j in range(13):
            distance += (i[j] - testCase[j])**2 
        distance = sqrt(distance)
        distance_table.append([distance,i[13]])
    distance_table.sort(key = dummy)
    for k in distance_table[0:K]:
        output_table.append(k[1])
    return output_table

def KNN_classifier(data,K):
    Accuracy = 0
    for m in range(1,11):
        #Splitting into training and testing sets of data
        Training = []
        Test = []
        Test_outcome = []
        Test_prediction = []
        split = 0.1
        Data_len = len(data)

        for x in range(Data_len):
            for y in range(14):
                data[x][y] = float(data[x][y])
            data[x][13] = 1 if data[x][13] else 0
            if ((x/Data_len) < m*split) and ((x/Data_len) > (m-1)*split):
                Test.append(data[x])
                Test_outcome.append(data[x][13])
            else:
                Training.append(data[x])
                
        Test_len = len(Test)

        for testCase in Test:
            Neighbours_values = KNearestNeighbours(Training,testCase,K)
            prediction = 1 if (Neighbours_values.count(1) > Neighbours_values.count(0)) else 0
            Test_prediction.append(prediction)

        Correct_predictions = 0
        for i in range(Test_len):
            if Test_prediction[i] == Test_outcome[i]:
                Correct_predictions += 1
        Accuracy += Correct_predictions*10/Test_len 

    
    return(Accuracy)

#loading data
with open('Heart_data_labels.csv', 'r') as csvFile:
    data = []
    reader = csv.reader(csvFile)
    for row in reader:
        data.append(row)
    data.pop(0)
csvFile.close()
print("K=1 Accuracty:",KNN_classifier(data,1))
print("K=3 Accuracty:",KNN_classifier(data,3))
print("K=5 Accuracty:",KNN_classifier(data,5))
print("K=9 Accuracty:",KNN_classifier(data,9))
print("K=15 Accuracty:",KNN_classifier(data,15))
# # Using sklearn library for KNN
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
df= pd.read_csv("Heart_data_labels.csv")
for i in [2,3,4]:
    df['Out'].replace(i,1,inplace = True)
X = np.array(df.drop(['Out'],1))
y = np.array(df['Out'])
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.3)

for i in [1,3,5,9,15]:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    clf.fit(X_train,y_train)
    Accuracy = clf.score(X_test,y_test)*100
    print("K:",i,"Accuracy: ",Accuracy)


