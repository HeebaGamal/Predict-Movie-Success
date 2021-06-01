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

class Controller:
  def __init__(self):
      pass

  def Draw_bar_graph(self, name, values_train, values_test):
      # data to plot
      n_groups = 6
      #values_train = (0, 0, 0, 0, 0, 0)
      #values_test = (0,0,0,0,0,0)

      # create plot
      fig, ax = plt.subplots()
      index = np.arange(n_groups)
      bar_width = 0.35
      opacity = 0.8

      rects1 = plt.bar(index, values_train, bar_width,
                       alpha=opacity,
                       color='b',
                       label='tarin')

      rects2 = plt.bar(index + bar_width, values_test, bar_width,
                       alpha=opacity,
                       color='g',
                       label='test')

      plt.xlabel('hyperparmertar')
      plt.ylabel('accuracy')
      plt.title(name)
      plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E', 'F'))
      plt.legend()

      plt.tight_layout()
      plt.show()

  def Data_preprocessing(self):
      # region Reading data
      # Loading data
      wd = os.getcwd()
      # Title,Year,Age,Rotten Tomatoes,Netflix,Hulu,Prime Video,Disney+,Type,Directors,Genres,Country,Language,Runtime,IMDb
      self.data = pd.read_csv(wd + "\\Movies_training_classification.csv")
      # Title             indx = -    x
      # Year              indx = 0
      # Age               indx = 1
      # Rotten tomatoes   indx = 2
      # Netflix           indx = 3    x
      # Hulu              indx = 4    x
      # Prime Video       indx = 5    x
      # Disney+           indx = 6    x
      # Type              indx = 7    x
      # Directors         indx = 8    x
      # Genres            indx = 9
      # Country           indx = 10
      # Language          indx = 11
      # Runtime           indx = 12
      # rate              indx = 13

      # endregion

      # region Missing data
      # Drop Age column as it has the least correlation with IMDb and most missing values & the title (doesn't affect the result)
      self.data = self.data.drop(['Title', 'Age'], axis=1)
      # Drop the rows that contain missing values
      self.data['Rotten Tomatoes'] = self.data['Rotten Tomatoes'].interpolate(method="pad")
      self.data.dropna(how='any', inplace=True)
      # endregion

      # region Encoding categorical data
      ord_enc = preprocessing.OrdinalEncoder()
      for i in ['Genres', 'Country', 'Language', 'rate', 'Rotten Tomatoes', 'Directors']:
          self.data[i] = ord_enc.fit_transform(self.data[[i]])
      # endregion
      # region Correlation
      # Get the correlation between the features
      corr = self.data.corr()
      '''
      # Top 50% Correlation training features with the Value
      top_feature = corr.index[abs(corr['rate'] >= 0.0)]

      # Correlation plot
      plt.subplots(figsize=(12, 8))
      top_corr = data[top_feature].corr()
      sns.heatmap(top_corr, annot=True)
      plt.show()'''
      # endregion
  def Normalization(self):
      # region Normalization
      X = self.data.iloc[:, :-1]
      Y = self.data.iloc[:, -1]
      X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X))
      # endregion
      # region Splitting data
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
      # endregion
      return X_train, X_test, Y_train, Y_test
  def Standarization(self):
      # region Standarization
      X = self.data.iloc[:, :-1]
      Y = self.data.iloc[:, -1]
      X = pd.DataFrame(StandardScaler().fit_transform(X))
      # endregion
      # region Splitting data
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
      # endregion
      return X_train, X_test, Y_train, Y_test
