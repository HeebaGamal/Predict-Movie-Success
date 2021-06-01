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
from sklearn.neighbors import KNeighborsClassifier
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

#region Model KNN

# we create an instance of KNN and fit out data.
n3_neig_uniform = KNeighborsClassifier(n_neighbors=3, weights='uniform').fit(X_train, Y_train)
n3_neig_distance = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X_train, Y_train)

n5_neig_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(X_train, Y_train)
n5_neig_distance = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(X_train, Y_train)

n9_neig_uniform = KNeighborsClassifier(n_neighbors=9, weights='uniform').fit(X_train, Y_train)
n9_neig_distance = KNeighborsClassifier(n_neighbors=9, weights='distance').fit(X_train, Y_train)




# title for the plots
titles = ['KNN with Neighbors_3 uniform',
          'KNN with Neighbors_3 distance',
          'KNN with Neighbors_5 uniform',
          'KNN with Neighbors_5 distance',
          'KNN with Neighbors_9 uniform',
          'KNN with Neighbors_9 distance'
          ]

for i, svm in enumerate((n3_neig_uniform, n3_neig_distance, n5_neig_uniform,
                         n5_neig_distance, n9_neig_uniform, n9_neig_distance)):
    filename = titles[i] + '.sav'
    pickle.dump(svm, open(filename, 'wb'))

#endregion
