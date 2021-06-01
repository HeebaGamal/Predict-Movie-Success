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
#endregion

#region Data_preprocessing
read_data = Controller()
read_data.Data_preprocessing()
#endregion

#region Normalization
X_train, X_test, Y_train, Y_test = read_data.Normalization()
#endregion


#region Model Logistic Regression
# Model 2 logistic regression
# hyperparameter 1 Regularization Parameter (inverse)
    # change 1 --> 0.00000000000000000001
    # change 2 --> 1 (default)
    # change 3 --> 99999999999999999999999
# hyperparameter 2 Max iterations
    # change 1 --> 10
    # change 2 --> 100 (default)
    # change 3 --> 100000000

lr_C1 = LogisticRegression(C=0.00000000000000000001).fit(X_train, Y_train)  # C is the inverse of regularization
lr_C2 = LogisticRegression(C=1).fit(X_train, Y_train)
lr_C3 = LogisticRegression(C=99999999999999999999999).fit(X_train, Y_train)

lr_MI1 = LogisticRegression(solver='saga', random_state=0).fit(X_train, Y_train)
lr_MI2 = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, Y_train)
lr_MI3 = LogisticRegression(solver='lbfgs', random_state=0).fit(X_train, Y_train)

titles = ['Logistic Regression Small C',
          'Logistic Regression Moderate C',
          'Logistic Regression Large C',
          'Logistic Regression Saga solver',
          'Logistic Regression Liblinear solver',
          'Logistic Regression Lbfgs solver']

for i, lr in enumerate((lr_C1, lr_C2, lr_C3, lr_MI1, lr_MI2, lr_MI3)):
    filename = titles[i]+'.sav'
    pickle.dump(lr, open(filename, 'wb'))



#endregion 
