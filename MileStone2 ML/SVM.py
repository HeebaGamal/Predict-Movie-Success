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

#region Model SVM
# model 1 SVM
# hyperparameter 1 Kernel
    # change 1 --> linear (1 vs all)
    # change 2 --> rbf
    # change 3 --> polynomial
# hyperparameter 2 C (regularization parameter) inverse
    # change 1 --> 0.0000000000000001
    # change 2 --> 1 (default)
    # change 3 --> 100000000000000000

# we create an instance of SVM and fit out data.
lin_svc = svm.LinearSVC().fit(X_train, Y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8).fit(X_train, Y_train)
poly_svc = svm.SVC(kernel='poly', degree=6).fit(X_train, Y_train)

svc_C1 = svm.LinearSVC(C=0.001).fit(X_train, Y_train)
svc_C2 = svm.LinearSVC(C=1).fit(X_train, Y_train)
svc_C3 = svm.LinearSVC(C=100).fit(X_train, Y_train)


# title for the plots
titles = ['SVC with Linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 6) kernel',
          'SVC with Small C',
          'SVC with Moderate C',
          'SVC with Large C']

for i, svm in enumerate((lin_svc, rbf_svc, poly_svc, svc_C1, svc_C2, svc_C3)):
    filename = titles[i] + '.sav'
    pickle.dump(svm, open(filename, 'wb'))

#endregion
