#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data= pd.read_csv("C://Users//deep//Desktop//Decodr//Case Studies_ Practice Files_ Reference Materials//Case Studies//Additional Solved Projects//Predict Mileage based on Technical Specifications of Automobile//auto-mpg.csv")


# In[4]:


data


# In[5]:


data.drop(['car name'], axis=1, inplace=True)
data.head()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data['horsepower'].unique()


# In[9]:


data = data[data.horsepower != '?']


# In[10]:


'?' in data


# In[11]:


data.shape


# In[12]:


data.corr()['mpg'].sort_values()


# In[13]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, center=0, cmap='rainbow')
plt.show()


# In[14]:


sns.countplot(data.cylinders, data=data, palette='rainbow')
plt.show()


# In[15]:


sns.countplot(data['model year'], palette='rainbow')
plt.show()


# In[16]:


sns.countplot(data.origin, palette='rainbow')
plt.show()


# In[17]:


sns.boxplot(y='mpg', x='cylinders', data=data, palette='rainbow')
plt.show()


# In[18]:


sns.boxplot(y='mpg', x='model year', data=data, palette='rainbow')
plt.show()


# In[19]:


X = data.iloc[:,1:].values
Y = data.iloc[:,0].values


# In[20]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regression = LinearRegression()
regression.fit(x_train,y_train)


# In[22]:


y_pred = regression.predict(x_test)


# In[23]:


print(regression.score(x_test, y_test))


# In[24]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X_poly,Y,test_size=0.3, random_state=0)

lin_regression = LinearRegression()
lin_regression.fit(x_train,y_train)

print(lin_regression.score(x_test, y_test))


# #Conclusion
# Accuracy score improves in the case of polynomial regression compared to the linear regression because it fits data much better. In this project, what we learned:
# 
# Loading the dataset
# Univariate analysis
# multivariate analysis
# Linear regression
# Polynomial Regression

# In[ ]:




