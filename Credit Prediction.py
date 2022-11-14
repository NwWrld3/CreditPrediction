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
from matplotlib.collections import LineCollection
import bezier
import fa2
from fa2 import ForceAtlas2
import jupyterthemes as jt
from jupyterthemes import get_themes
from jupyterthemes.stylefx import set_nb_theme
set_nb_theme('monokai')


# In[2]:


#load data and print dataframe
df = pd.read_excel('/Users/Vincent/Library/CloudStorage/OneDrive-Personal/Uni/Psychology/Thesis/CREDIT TEST.xlsx')
print (df)


# In[3]:


#assign labels to dataframe columns
df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'IQ', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1','C2','C3','C4','C5','C6','C7','C8','C9','Credit']


# In[4]:


#print dataframe head with new labels
df.head()


# In[5]:


#print shape of the dataframe
df.shape


# In[6]:


#assign values to X and y and print them
X=df[['A1', 'A2', 'A3', 'A4', 'A5', 'IQ', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1', 'C2','C3', 'C4', 'C5','C6','C7','C8','C9']]
y=df[['Credit']]
print(X)
print(y)


# In[7]:


#split the data into training and testing sections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)


# In[8]:


#print correlations to credit
correlation_matrix = df.corr()
correlation_matrix['Credit']


# In[9]:


#print heatmap of correlations
sns.set(rc={'figure.figsize':(6,4)}) 
sns.set(rc={'axes.facecolor':'black','figure.facecolor':'black','grid.alpha': 0.2,'axes.edgecolor':'white','text.color':'white','xtick.labelsize': '9.0','ytick.labelsize': '9.0','xtick.color':'r','ytick.color':'r'})
sns.heatmap(df.corr(), cmap="inferno")


# In[10]:


from sklearn import preprocessing


# In[11]:


#normalise x values
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)

df = pd.DataFrame(df)


# In[12]:


corr = df.corr()


# In[13]:


print(corr)


# In[14]:


#print correlations to credit
correlation_matrix = df.corr()
correlation_matrix["Credit"]


# In[29]:


sns.set(rc={'figure.figsize':(7,5)})
sns.set(rc={'axes.facecolor':'black','figure.facecolor':'black','grid.alpha': 0.2,'axes.edgecolor':'grey','text.color':'white','xtick.labelsize': '9.0','ytick.labelsize': '9.0','xtick.color':'r','ytick.color':'r'})
sns.scatterplot(x='C9', y='Credit', data=df, edgecolor='b', s=8, alpha=0.9, legend='full')
plt.title('C9 and Credit')
plt.xlabel('C9')
plt.ylabel('Credit')
plt.savefig('C9andCredit')


# In[17]:


import networkx as nx


# In[18]:


G = nx.Graph()


# In[19]:


A = np.array(corr)


# In[20]:


G = nx.from_numpy_array(A)


# In[21]:


G.edges(data=True)


# In[22]:


G.remove_edges_from(nx.selfloop_edges(G))


# In[23]:


import bezier
import networkx as nx
import numpy as np

def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=80, polarity='random'):
    # Get nodes into np array
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(edges[:,0])+np.vectorize(hash)(edges[:,1]),2)==0,-1,1)
    
    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse = True)
    coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:,0,:]
    coords_node2 = coords[:,1,:]
    
    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:,0] > coords_node2[:,0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]
    
    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:,1]-coords_node1[:,1])/(coords_node2[:,0]-coords_node1[:,0])
    m2 = -1/m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist/np.sqrt(1+m1**2)
    v1 = np.array([np.ones(l),m1])
    coords_node1_displace = coords_node1 + (v1*t1).T
    coords_node2_displace = coords_node2 - (v1*t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist/np.sqrt(1+m2**2)
    v2 = np.array([np.ones(len(edges)),m2])
    coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:,i,:].T
        curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0,1,bezier_precision)).T)
      
    # Return an array of these curves
    curves = np.array(curveplots)
    return curves


# In[24]:


forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=0.5,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=0.5,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=1.0,
                        strongGravityMode=True,
                        gravity=0.1,

                        # Log
                        verbose=True)

positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=5)
curves = curved_edges(G, positions)
lc = LineCollection(curves, color=['indigo','lightgreen','white','lightblue','blue'], alpha=0.1)
plt.figure(figsize=(20,20))
nx.draw_networkx_labels(G, positions, font_size=10)
nx.draw_networkx_nodes(G, positions, node_size=0.1, node_color="orange",alpha=0.3)

plt.gca().add_collection(lc)
plt.axis('off')
plt.show()


# In[25]:


fig = plt.figure()
nx.draw(G, with_labels=True, font_color='grey',edge_color=['orange','lightgreen','lightblue'], node_color='skyblue',node_size=0.1,width=0.1,alpha=0.5)
fig.set_facecolor("Black")


# In[26]:


G.number_of_edges()


# In[27]:


G.number_of_nodes()


# In[ ]:





# In[ ]:





# In[ ]:




