#region Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, tree
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, label, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import os
import pickle

from ScriptControler import Controller

c = Controller()
bar = [[0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]
#endregion

#region Data_preprocessing
read_data = Controller()
read_data.Data_preprocessing()
#endregion

#region Normalization
X_train, X_test, Y_train, Y_test = read_data.Normalization()
#endregion

#region Model SVM
titles = ['SVC with Linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 6) kernel',
          'SVC with Small C',
          'SVC with Moderate C',
          'SVC with Large C']
i=0
for title in titles:
    SVM_file = open(title + '.sav', 'rb')
    svm = pickle.load(SVM_file)

    predictions = svm.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    bar[0][i] = accuracy
    print(title + ' train accuracy : ' + str(accuracy))

    predictions = svm.predict(X_test)
    bar[1][i] = accuracy
    accuracy = np.mean(predictions == Y_test)
    print(title + ' test accuracy : ' + str(accuracy))

    i+=1
    print()

    SVM_file.close()
c.Draw_bar_graph('SVM', np.array(bar[0]), np.array(bar[1]))

#endregion

#region Model Logistic Regression
titles = ['Logistic Regression Small C',
          'Logistic Regression Moderate C',
          'Logistic Regression Large C',
          'Logistic Regression Saga solver',
          'Logistic Regression Liblinear solver',
          'Logistic Regression Lbfgs solver']
i=0
for title in titles:
    LR_file = open(title + '.sav', 'rb')
    lr = pickle.load(LR_file)

    predictions = lr.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    bar[0][i] = accuracy
    print(title + ' train accuracy : ' + str(accuracy))

    predictions = lr.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    bar[1][i] = accuracy
    print(title + ' test accuracy : ' + str(accuracy))

    i+=1
    print()

    LR_file.close()
c.Draw_bar_graph('Logistic Regression', np.array(bar[0]), np.array(bar[1]))
#endregion

#region Gradient Boosting
titles = ['Gradient Boosting small number of estimators',
          'Gradient Boosting moderate number of estimators',
          'Gradient Boosting large number of estimators',
          'Gradient Boosting small learning rate',
          'Gradient Boosting moderate learning rate',
          'Gradient Boosting large learning rate']
i = 0
for title in titles:
    GB_file = open(title + '.sav', 'rb')
    gb = pickle.load(GB_file)

    predictions = gb.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    bar[0][i] = accuracy
    print(title + ' train accuracy : ' + str(accuracy))

    predictions = gb.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    bar[1][i] = accuracy
    print(title + ' test accuracy : ' + str(accuracy))

    i+=1
    print()

    GB_file.close()

c.Draw_bar_graph('Gradient Boosting', np.array(bar[0]), np.array(bar[1]))

#endregion

#region KNN
titles = ['KNN with Neighbors_3 uniform',
          'KNN with Neighbors_3 distance',
          'KNN with Neighbors_5 uniform',
          'KNN with Neighbors_5 distance',
          'KNN with Neighbors_9 uniform',
          'KNN with Neighbors_9 distance'
          ]
i = 0
for title in titles:
    GB_file = open(title + '.sav', 'rb')
    gb = pickle.load(GB_file)

    predictions = gb.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    bar[0][i] = accuracy
    print(title + ' train accuracy : ' + str(accuracy))

    predictions = gb.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    bar[1][i] = accuracy
    print(title + ' test accuracy : ' + str(accuracy))

    i+=1
    print()

    GB_file.close()

c.Draw_bar_graph('KNN', np.array(bar[0]), np.array(bar[1]))

#endregion

#region Standarization
X_train, X_test, Y_train, Y_test  = read_data.Standarization()
#endregion

#region Adaboost Decision tree
titles = ['ADT small number of estimators',
          'ADT moderate number of estimators',
          'ADT large number of estimators',
          'ADT small learning rate',
          'ADT moderate learning rate',
          'ADT large learning rate']

i = 0
for title in titles:
    ADT_file = open(title + '.sav', 'rb')
    adt = pickle.load(ADT_file)

    predictions = adt.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    bar[0][i] = accuracy
    print(title + ' train accuracy : ' + str(accuracy))

    predictions = adt.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    bar[1][i] = accuracy
    print(title + ' test accuracy : ' + str(accuracy))

    i+=1
    print()

    ADT_file.close()

c.Draw_bar_graph('Adaboost Decision tree', np.array(bar[0]), np.array(bar[1]))

#endregion

