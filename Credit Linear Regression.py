#ANN for credit prediction
#Authors:
#Vincent Weidlich(vincentweidlich89@gmail.com)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import rcParams
import sklearn
from sklearn import neighbors, datasets
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
import jupyterthemes as jt
from jupyterthemes import get_themes
from jupyterthemes.stylefx import set_nb_theme
set_nb_theme('monokai')


# In[2]:


#load data and print dataframe
df = pd.read_excel('/Users/Vincent/Library/CloudStorage/OneDrive-Personal/Uni/Psychology/Thesis/CREDIT TEST.xlsx')
print (df)


# In[5]:


#assign labels to dataframe columns
df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'IQ', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1','C2','C3','C4','C5','C6','C7','C8','C9','Credit']


# In[6]:


#assign values to X and y and print them
X=df[['A1', 'A2', 'A3', 'A4', 'A5', 'IQ', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1', 'C2','C3', 'C4', 'C5','C6','C7','C8','C9']]
y=df[['Credit']]
print(X)
print(y)


# In[7]:


#normalise x values
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)

df = pd.DataFrame(df)


# In[8]:


X = df['C3'].values.reshape(-1,1)
y = df['Credit'].values


# In[9]:


ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)


# In[10]:


r2 = model.score(X, y)


# In[11]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='red', label='Regression Model', alpha=0.5)
ax.scatter(X, y, edgecolor='grey', facecolor='grey', alpha=0.9, label='Credit Score Predictor',s=9,color='r')
ax.set_ylabel('Credit Score', fontsize=10,color='r',alpha=0.5)
ax.set_xlabel('C3', fontsize=10, color='r', alpha=0.5)
ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

ax.legend(facecolor='grey', fontsize=8)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18, color='r', alpha=0.5)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.9)
fig.tight_layout()


# In[12]:


X = df['C4'].values.reshape(-1,1)
y = df['Credit'].values


# In[13]:


ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)


# In[14]:


r2 = model.score(X, y)


# In[15]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='red', label='Regression Model', alpha=0.5)
ax.scatter(X, y, edgecolor='grey', facecolor='grey', alpha=0.9, label='Credit Score Predictor',s=9,color='r')
ax.set_ylabel('Credit Score', fontsize=10,color='r',alpha=0.5)
ax.set_xlabel('C4', fontsize=10, color='r', alpha=0.5)
ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

ax.legend(facecolor='grey', fontsize=8)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18, color='r', alpha=0.5)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.9)
fig.tight_layout()


# In[16]:


X = df['C6'].values.reshape(-1,1)
y = df['Credit'].values


# In[17]:


ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)


# In[18]:


r2 = model.score(X, y)


# In[19]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='red', label='Regression Model', alpha=0.5)
ax.scatter(X, y, edgecolor='grey', facecolor='grey', alpha=0.9, label='Credit Score Predictor',s=9,color='r')
ax.set_ylabel('Credit Score', fontsize=10,color='r',alpha=0.5)
ax.set_xlabel('C6', fontsize=10, color='r', alpha=0.5)
ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

ax.legend(facecolor='grey', fontsize=8)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18, color='r', alpha=0.5)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.9)
fig.tight_layout()


# In[20]:


X = df['C8'].values.reshape(-1,1)
y = df['Credit'].values


# In[21]:


ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)


# In[22]:


r2 = model.score(X, y)


# In[23]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='red', label='Regression Model', alpha=0.5)
ax.scatter(X, y, edgecolor='grey', facecolor='grey', alpha=0.9, label='Credit Score Predictor',s=9,color='r')
ax.set_ylabel('Credit Score', fontsize=10,color='r',alpha=0.5)
ax.set_xlabel('C8', fontsize=10, color='r', alpha=0.5)
ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

ax.legend(facecolor='grey', fontsize=8)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18, color='r', alpha=0.5)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.9)
fig.tight_layout()


# In[24]:


X = df['C9'].values.reshape(-1,1)
y = df['Credit'].values


# In[25]:


ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)


# In[26]:


r2 = model.score(X, y)


# In[27]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='red', label='Regression Model', alpha=0.5)
ax.scatter(X, y, edgecolor='grey', facecolor='grey', alpha=0.9, label='Credit Score Predictor',s=9,color='r')
ax.set_ylabel('Credit Score', fontsize=10,color='r',alpha=0.5)
ax.set_xlabel('C9', fontsize=10, color='r', alpha=0.5)
ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

ax.legend(facecolor='grey', fontsize=8)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18, color='r', alpha=0.5)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.9)
fig.tight_layout()


# In[28]:


features = ['C9']
target = 'Credit'

X = df[features].values.reshape(-1, len(features))
y = df[target].values


# In[29]:


print(X.shape)
print(y.shape)


# In[30]:


from sklearn import linear_model

ols = linear_model.LinearRegression()
model = ols.fit(X, y)


# In[31]:


model.coef_


# In[32]:


model.intercept_


# In[33]:


model.score(X, y)


# In[34]:


x_pred = np.array([0.2]) #input value predictor
x_pred = x_pred.reshape(-1, len(features))  # preprocessing required by scikit-learn functions


# In[35]:


model.predict(x_pred) #predict credit score with one variable


# In[36]:


x_pred = np.array([0.2, 0.3])#multiple input values of predictor
x_pred = x_pred.reshape(-1, len(features))  # preprocessing required by scikit-learn functions


# In[37]:


model.predict(x_pred) #predict credit score with multiple instances of the same variable


# In[38]:


x_pred = np.linspace(0,0.5, 40)            
x_pred = x_pred.reshape(-1, len(features))  # preprocessing required by scikit-learn functions

y_pred = model.predict(x_pred)


# In[39]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(7, 3.5))

ax.plot(x_pred, y_pred,  color='red', label='Regression model',alpha=0.5)





ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

ax.set_title('$R^2= %.2f$' % r2, fontsize=18, color='r', alpha=0.5)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.9)
fig.tight_layout()


ax.scatter(X, y, edgecolor='grey', facecolor='grey', alpha=0.9, label='Credit Score Predictor',s=9,color='r')
ax.set_ylabel('Credit Score', fontsize=10,color='r',alpha=0.5)
ax.set_xlabel('C9', fontsize=10, color='r', alpha=0.5)
ax.legend(facecolor='grey', fontsize=8)
ax.text(0.55, 0.15, '$y = %.2f x_1 - %.2f $' % (model.coef_[0], abs(model.intercept_)), fontsize=9, transform=ax.transAxes)


# In[40]:


ols = linear_model.LinearRegression(fit_intercept=False)
model = ols.fit(X, y)


# In[41]:


model.intercept_


# In[42]:


features = ['C3','C5','C4','C1','B6','B2','IQ','C6','C7']
target = 'Credit'

X = df[features].values.reshape(-1, len(features))
y = df[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)


# In[43]:


model.coef_


# In[44]:


model.intercept_


# In[45]:


model.score(X, y) #Print model accuracy


# In[ ]:




