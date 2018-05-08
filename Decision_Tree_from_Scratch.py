import numpy as np
import pandas as pd
import random
from sklearn import model_selection
df= pd.read_csv("Heart_data_labels.csv")
for i in [2,3,4]:
    df['Out'].replace(i,1,inplace = True)
df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Trestbps"] = (df["Trestbps"] - df["Trestbps"].min()) / (df["Trestbps"].max() - df["Trestbps"].min())
df["Chol"] = (df["Chol"] - df["Chol"].min()) / (df["Chol"].max() - df["Chol"].min())
df["thalach"] = (df["thalach"] - df["thalach"].min()) / (df["thalach"].max() - df["thalach"].min())
X = np.array(df.drop("Out",axis = 1))
y = np.transpose(np.array([df["Out"]]))
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)
data = np.array(df)
header = list(df.columns.values)

def unique_vals(rows, col): #returns all the unique values of a given column
    return set([row[col] for row in rows])

def class_counts(rows): #return the number of classes in the data
    counts = {}  
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question: #Templete for the question (whether it is greater than the threshold value)
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, example):
        val = example[self.column]
        return val >= self.value

def partition(rows, question): #Split the data into True and False brances of the node
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows): #Metric for a questions
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty): #For valuing the split based on Gini value
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows): #Finding the best split , i.e, figuring the best question for the split
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf: # end point of a tree
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node: # Intermediate node of a tree
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows): #Main function for decision tree
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

Accuracy = 0.0
for i in np.arange(0.0, 1.0, 0.1):
    Test_data = data[int(i*len(data)):int((i+0.1)*len(data)),:]
    Train_data = np.array([j for j in data.tolist() if j not in Test_data.tolist()])
    solution = build_tree(Train_data)
    correct_count = 0.0
    for row in Test_data: 
        [(k, v)] = classify(row, solution).items()
        if k == row[-1]:
            correct_count += 1
    Accuracy += (correct_count/len(Test_data))*0.1 #10-fold cross validation
Accuracy *= 100

print(Accuracy)

# # Using sklearn

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction = dtree.predict(X_test)

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
