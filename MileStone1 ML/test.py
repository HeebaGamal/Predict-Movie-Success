#region Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import label
from sklearn import preprocessing
import pickle
import os
#endregion

#region Read Data
#Loading data
# Title,Year,Age,Rotten Tomatoes,Netflix,Hulu,Prime Video,Disney+,Type,Directors,Genres,Country,Language,Runtime,IMDb
wd = os.getcwd()
data = pd.read_csv(wd+"\\Movies_training.csv")

# Title 11743
# Year 11743            indx = 0    num     -0.07097082909007243
# Age 5814
# Rotten Tomatoes 5055  indx = 1    percentage
# Netflix 11743         indx = 2    bool    0.057828135006535025
# Hulu 11743            indx = 3    bool    -0.005814027869158413
# Prime Video 11743     indx = 4    bool    -0.07696780805599245
# Disney+ 11743         indx = 5    bool    0.05114274183651775
# Type 11743            indx = 6    bool    NAN
# Directors 11202       indx = 7    list of names
# Genres 11518          indx = 8    list of names
# Country 11392         indx = 9    list of names
# Language 11291        indx = 10   list of names
# Runtime 11272         indx = 11   num     0.12955211666633556
# IMDB 11236            indx = 12   num

#endregion

#region Missing data
# Drop Age column as it has the least correlation with IMDb and most missing values & the title (doesn't affect the result)
data = data.drop(['Title'], axis=1)
# Drop the rows that contain missing values
data.dropna(how='any', inplace=True, subset=['Runtime', 'Genres', 'IMDb','Language', 'Country', 'Directors'])
data['Rotten Tomatoes'] = data['Rotten Tomatoes'].interpolate(method="pad")
data['Age'] = data['Age'].interpolate(method="pad")
#endregion

#region OneHot Encoder

# mapping non numerical values to number
label_encoder = label.LabelEncoder()
data['Age'] = label_encoder.fit_transform(data['Age'])
onehot_encoder = preprocessing.OneHotEncoder()
sq = onehot_encoder.fit_transform(data[['Age']]).toarray()
col = 1
sq = np.array(sq)

for i in range(sq.shape[1]):
    data.insert(col, str(i), sq[:, i])
    col += 1
data = data.drop(['Age'], axis=1)
#endregion

#region Ordinal Encoder
ord_enc = preprocessing.OrdinalEncoder()
for i in ['Genres', 'Rotten Tomatoes', 'Language', 'Country', 'Directors']:
    data[i] = ord_enc.fit_transform(data[[i]])
#endregion

#region Correlation
# Get the correlation between the features
corr = data.corr()

# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['IMDb'] >= 0.0)]

plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
#endregion

#region Normalization
data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data))
#endregion

#region Splitting data
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
#endregion

#region Linear Regression
linear_file = open('Linear Regression.sav', 'rb')
cls = pickle.load(linear_file)

prediction = cls.predict(X)
print('Mean Square Error test Linear ', metrics.mean_squared_error(np.asarray(Y), prediction))
print('R2 Score test Linear ', metrics.r2_score(np.asarray(Y), prediction))
print()
#endregion

#region Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X = poly_features.fit_transform(X)

Poly_file = open('Polynomial Regression.sav', 'rb')
Polynomial_Model = pickle.load(Poly_file)

prediction = Polynomial_Model.predict(X)
print('Mean Square Error test Polynomial ', metrics.mean_squared_error(np.asarray(Y), prediction))
print('R2 Score test Polynomial ', metrics.r2_score(np.asarray(Y), prediction))
#endregion
