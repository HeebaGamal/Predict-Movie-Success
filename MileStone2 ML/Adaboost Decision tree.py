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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import preprocessing
import os
import pickle
from ScriptControler import Controller
#endregion

#region Data_preprocessing
read_data = Controller()
read_data.Data_preprocessing()
#endregion

#region Normalization
X_train, X_test, Y_train, Y_test = read_data.Standarization()
#endregion



#region Model Adaboost Decision tree
# Model 3
# hyperparameter 1 n_estimators
    # change 1 --> 10
    # change 2 --> 100
    # change 3 --> 1000
# hyperparameter 2 learning rate
    # change 1 --> 0.001
    # change 2 --> 0.1
    # change 3 --> 1.4

adt_NE1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME.R", n_estimators=10,
                             random_state=0).fit(X_train, Y_train)
adt_NE2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME.R", n_estimators=100,
                             random_state=0).fit(X_train, Y_train)
adt_NE3 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME.R", n_estimators=1000,
                             random_state=0).fit(X_train, Y_train)

adt_LR1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), algorithm="SAMME.R", n_estimators=100,
                             learning_rate=0.01, random_state=0).fit(X_train, Y_train)
adt_LR2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), algorithm="SAMME.R", n_estimators=100,
                             learning_rate=0.17, random_state=0).fit(X_train, Y_train)
adt_LR3 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), algorithm="SAMME.R", n_estimators=100,
                             learning_rate=1.4, random_state=0).fit(X_train, Y_train)

titles = ['ADT small number of estimators',
          'ADT moderate number of estimators',
          'ADT large number of estimators',
          'ADT small learning rate',
          'ADT moderate learning rate',
          'ADT large learning rate']

for i, adt in enumerate((adt_NE1, adt_NE2, adt_NE3, adt_LR1, adt_LR2, adt_LR3)):
    filename = titles[i]+'.sav'
    pickle.dump(adt, open(filename, 'wb'))

#endregion
