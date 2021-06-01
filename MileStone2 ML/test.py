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
#endregion

#region Reading data
#Loading data
wd = os.getcwd()
# Title,Year,Age,Rotten Tomatoes,Netflix,Hulu,Prime Video,Disney+,Type,Directors,Genres,Country,Language,Runtime,IMDb
data = pd.read_csv(wd+"\\Movies_testing_classification.csv")
# Title             indx = -
# Year              indx = 0
# Age               indx = 1
# Rotten tomatoes   indx = 2
# Netflix           indx = 3
# Hulu              indx = 4
# Prime Video       indx = 5
# Disney+           indx = 6
# Type              indx = 7
# Directors         indx = 8
# Genres            indx = 9
# Country           indx = 10
# Language          indx = 11
# Runtime           indx = 12
# rate              indx = 13

#endregion

#region Missing data
# Drop Age column as it has the least correlation with IMDb and most missing values & the title (doesn't affect the result)
data = data.drop(['Title', 'Age'], axis=1)
# Drop the rows that contain missing values
data['Rotten Tomatoes'] = data['Rotten Tomatoes'].interpolate(method="pad")
data['Year'] = data['Year'].interpolate(method="pad")
data['Netflix'] = data['Netflix'].interpolate(method="pad")
data['Hulu'] = data['Hulu'].interpolate(method="pad")
data['Prime Video'] = data['Prime Video'].interpolate(method="pad")
data['Disney+'] = data['Disney+'].interpolate(method="pad")
data['Type'] = data['Type'].interpolate(method="pad")
data['Directors'] = data['Directors'].interpolate(method="pad")
data['Genres'] = data['Genres'].interpolate(method="pad")
data['Country'] = data['Country'].interpolate(method="pad")
data['Language'] = data['Language'].interpolate(method="pad")
data['Runtime'] = data['Runtime'].interpolate(method="pad")

#endregion

#region Encoding categorical data
ord_enc = preprocessing.OrdinalEncoder()
for i in ['Genres', 'Country', 'Language', 'rate', 'Rotten Tomatoes', 'Directors']:
    data[i] = ord_enc.fit_transform(data[[i]])
#endregion

#region Correlation
# Get the correlation between the features
corr = data.corr()
'''
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['rate'] >= 0.0)]

# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()'''
#endregion

#region Normalization
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X))
#endregion

#region Model SVM
titles = ['SVC with Linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 6) kernel',
          'SVC with Small C',
          'SVC with Moderate C',
          'SVC with Large C']

for title in titles:
    SVM_file = open(title + '.sav', 'rb')
    svm = pickle.load(SVM_file)

    predictions = svm.predict(X)
    accuracy = np.mean(predictions == Y)
    print(title + ' test accuracy : ' + str(accuracy))

    print()

    SVM_file.close()
#endregion

#region Model Logistic Regression
titles = ['Logistic Regression Small C',
          'Logistic Regression Moderate C',
          'Logistic Regression Large C',
          'Logistic Regression Saga solver',
          'Logistic Regression Liblinear solver',
          'Logistic Regression Lbfgs solver']

for title in titles:
    LR_file = open(title + '.sav', 'rb')
    lr = pickle.load(LR_file)

    predictions = lr.predict(X)
    accuracy = np.mean(predictions == Y)
    print(title + ' test accuracy : ' + str(accuracy))

    print()

    LR_file.close()
#endregion

#region Gradient Boosting
titles = ['Gradient Boosting small number of estimators',
          'Gradient Boosting moderate number of estimators',
          'Gradient Boosting large number of estimators',
          'Gradient Boosting small learning rate',
          'Gradient Boosting moderate learning rate',
          'Gradient Boosting large learning rate']

for title in titles:
    GB_file = open(title + '.sav', 'rb')
    gb = pickle.load(GB_file)

    predictions = gb.predict(X)
    accuracy = np.mean(predictions == Y)
    print(title + ' test accuracy : ' + str(accuracy))

    print()

    GB_file.close()
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

    predictions = gb.predict(X)
    accuracy = np.mean(predictions == Y)
    print(title + ' test accuracy : ' + str(accuracy))

    i+=1
    print()

    GB_file.close()


#endregion



#region Standarization
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X = pd.DataFrame(StandardScaler().fit_transform(X))
#endregion

#region Adaboost Decision tree
titles = ['ADT small number of estimators',
          'ADT moderate number of estimators',
          'ADT large number of estimators',
          'ADT small learning rate',
          'ADT moderate learning rate',
          'ADT large learning rate']

for title in titles:
    ADT_file = open(title + '.sav', 'rb')
    adt = pickle.load(ADT_file)

    predictions = adt.predict(X)
    accuracy = np.mean(predictions == Y)
    print(title + ' test accuracy : ' + str(accuracy))

    print()

    ADT_file.close()
#endregion



