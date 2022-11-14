#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import jupyterthemes as jt
import pandas as pd
import ann_visualizer
from ann_visualizer.visualize import ann_viz
import sklearn
from sklearn import neighbors, datasets
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from keras.models import Sequential
from jupyterthemes import get_themes
from jupyterthemes.stylefx import set_nb_theme
set_nb_theme('monokai')


# In[2]:


df = pd.read_excel('/Users/Vincent/Library/CloudStorage/OneDrive-Personal/Uni/Psychology/Thesis/CREDIT TEST.xlsx')


# In[3]:


df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'IQ', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1','C2','C3','C4','C5','C6','C7','C8','C9','Credit']


# In[4]:


X=df[['C3','C5','C4','C1','B6','B2','IQ','C6','C7']]
y=df[['Credit']]


# In[5]:


# Splitting training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

# printing the input and output shapes
len(X_train), len(X_test)


# In[27]:


# fixing random state
tf.random.set_seed(42)

model = tf.keras.Sequential([tf.keras.layers.InputLayer(
    input_shape=9),
  tf.keras.layers.Dense(100, activation = tf.keras.activations.relu),
  tf.keras.layers.Dense(300, activation = tf.keras.activations.softmax),
  tf.keras.layers.Dense(20, activation = tf.keras.activations.relu),
  tf.keras.layers.Dense(1000, activation = tf.keras.activations.softmax),
  tf.keras.layers.Dense(200, activation = tf.keras.activations.relu),
  tf.keras.layers.Dense(2000, activation = tf.keras.activations.sigmoid),
  tf.keras.layers.Dense(100, activation = tf.keras.activations.relu),
                                   
  tf.keras.layers.Dense(1)
])

# compiling the model
model.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])

# Training the model
model.fit(X_train, y_train, epochs=10000, verbose=0)


# In[28]:


preds_credit = model.predict(X_test)


# In[29]:


# Evaluating the model
print('R score is :', r2_score(y_test, preds_credit))


# In[30]:


# fitting the size of the plot
fig = plt.figure()
plt.figure(figsize=(16, 8),facecolor='black')
ax = plt.gca()
ax.set_title('ANN', fontsize=18, color='r', alpha=0.5)

ax.patch.set_facecolor('black')
ax.grid(color='r', linewidth=0.3, alpha=0.5)
ax.tick_params(colors='r', size=0.5,)

# plotting the graphs
plt.plot([i for i in range(len(y_test))],y_test, label="Actual values", c='r')
plt.plot([i for i in range(len(y_test))],preds_credit, label="Predicted values", c='g')

# showing the plotting
plt.legend()
ax.legend(facecolor='grey', fontsize=10)
plt.show()

plt.show()


# In[ ]:




