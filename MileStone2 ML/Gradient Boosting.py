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
X_train, X_test, Y_train, Y_test = read_data.Normalization()
#endregion

#region Model Gradient Boosting
# Model 4
# hyperparameter 1 n_estimators
    # change 1 --> 10
    # change 2 --> 100
    # change 3 --> 1000
# hyperparameter 2 learning rate
    # change 1 --> 0.01
    # change 2 --> 0.17
    # change 3 --> 1.4

gb_NE1 = GradientBoostingClassifier(max_depth=5, n_estimators=10, random_state=0).fit(X_train, Y_train)
gb_NE2 = GradientBoostingClassifier(max_depth=5, n_estimators=100, random_state=0).fit(X_train, Y_train)
gb_NE3 = GradientBoostingClassifier(max_depth=5, n_estimators=1000, random_state=0).fit(X_train, Y_train)

gb_LR1 = GradientBoostingClassifier(max_depth=5, n_estimators=100, learning_rate=0.01,
                                    random_state=0).fit(X_train, Y_train)
gb_LR2 = GradientBoostingClassifier(max_depth=5, n_estimators=100, learning_rate=0.17,
                                    random_state=0).fit(X_train, Y_train)
gb_LR3 = GradientBoostingClassifier(max_depth=5, n_estimators=100, learning_rate=1.4,
                                    random_state=0).fit(X_train, Y_train)

titles = ['Gradient Boosting small number of estimators',
          'Gradient Boosting moderate number of estimators',
          'Gradient Boosting large number of estimators',
          'Gradient Boosting small learning rate',
          'Gradient Boosting moderate learning rate',
          'Gradient Boosting large learning rate']

for i, lr in enumerate((gb_NE1, gb_NE2, gb_NE3, gb_LR1, gb_LR2, gb_LR3)):
    filename = titles[i]+'.sav'
    pickle.dump(lr, open(filename, 'wb'))


#endregion
