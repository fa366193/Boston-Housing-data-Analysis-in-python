#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score as cvs


# In[3]:


#creating dataset table
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv("Desktop/housing.csv", delimiter=r'\s+', names=column_names)
dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


#removing variables 'ZN' and 'CHAS' from data
dataset = dataset.drop(['ZN', 'CHAS'], axis=1)


# In[7]:


#checking null values
dataset.isnull().sum()


# In[8]:


#Plotting boxplots to see if there are any outliers in our data (considering data betwen 25th and 75th percentile as non outlier)
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
for i in dataset.columns:
  sns.boxplot(y=i, data=dataset, ax=ax[index])
  index +=1
plt.tight_layout(pad=0.4)
plt.show()


# In[9]:


#checking percentage/ amount of outliers
for i in dataset.columns:
  dataset.sort_values(by=i, ascending=True, na_position='last')
  q1, q3 = np.nanpercentile(dataset[i], [25,75])
  iqr = q3-q1
  lower_bound = q1-(1.5*iqr)
  upper_bound = q3+(1.5*iqr)
  outlier_data = dataset[i][(dataset[i] < lower_bound) | (dataset[i] > upper_bound)] #creating a series of outlier data
  perc = (outlier_data.count()/dataset[i].count())*100
  print('Outliers in %s is %.2f%% with count %.f' %(i, perc, outlier_data.count()))


# In[10]:


if i == 'B':
  outlierDataB_index = outlier_data.index
  outlierDataB_LB = dataset[i][(dataset[i] < lower_bound)]
  outlierDataB_UB = dataset[i][(dataset[i] > upper_bound)]
elif i == 'CRIM':
  outlierDataCRIM_index = outlier_data.index
  outlierDataCRIM_LB = dataset[i][(dataset[i] < lower_bound)]
  outlierDataCRIM_UB = dataset[i][(dataset[i] > upper_bound)]
elif i == 'MEDV':
  lowerBoundMEDV = lower_bound
  upperBoundMEDV = upper_bound


# In[11]:


dataset2 = dataset.copy()


# In[13]:


dataset3 = dataset2.copy()


# In[14]:


#replacing remaning outliers by mean
for i in dataset.columns:
  dataset.sort_values(by=i, ascending=True, na_position='last')
  q1, q3 = np.nanpercentile(dataset[i], [25,75])
  iqr = q3-q1
  lower_bound = q1-(1.5*iqr)
  upper_bound = q3+(1.5*iqr)
  mean = dataset3[i].mean()
  if i != 'MEDV':
    dataset3.loc[dataset3[i] < lower_bound, [i]] = mean
    dataset3.loc[dataset3[i] > upper_bound, [i]] = mean
  else:
    dataset3.loc[dataset3[i] < lower_bound, [i]] = mean
    dataset3.loc[dataset3[i] > upper_bound, [i]] = 50


# In[15]:


dataset3.describe()


# In[16]:


#independent variable(X) and dependent variable(Y)
X = dataset3.iloc[:, :-1]
Y = dataset3.iloc[:, 11]


# In[18]:


#Ploting heatmap using pearson correlation among independent variables
plt.figure(figsize=(8, 8))
ax = sns.heatmap(X.corr(method='pearson').abs(), annot=True, square=True)
plt.show()


# In[ ]:




