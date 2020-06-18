#!/usr/bin/env python
# coding: utf-8

# In[79]:


# In this code, I have implemented a Decision Tree Algorithm on a popular dataset named CAR DATASET. I will implement it with
# splitting the data using train_test_split 70:30 strategy. Post implementation, I will visualise the Decision Tree.


# In[80]:


# Load Info About the Dataset
f = open('D:/car.names')
for i in f:
    print(i)


# In[81]:


# Load Basic Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[82]:


# Load Data
df = pd.read_csv('D:/car.data', header=None,
                names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','CAR'])
df.head()


# In[83]:


df.shape # Total 1728 records


# In[84]:


# Let us have a look on the output : Shows 4 classes viz unacceptable, acceptable, very good and good
df['CAR'].unique()


# In[85]:


sb.countplot(df['CAR'])


# In[86]:


# Shows about 70% cars are unacceptable, about 400 carrs are acceptable. Very few comes under Very Good and Good Category


# In[87]:


df.head(2)


# In[88]:


# To fit data in a model, input values needs to be numeric. Lets check each column and encode the values in numeric form,
# if needed
df['buying'].unique()
df['buying'] = df['buying'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})


# In[89]:


df['maint'].unique()
df['maint'] = df['maint'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})


# In[90]:


df['doors'].unique()
df['doors'] = df['doors'].replace({'2':0, '3':1, '4':2, '5more':3})


# In[91]:


df['persons'].unique()
df['persons'] = df['persons'].replace({'2':0, '4':1, 'more':2})


# In[92]:


df['lug_boot'].unique()
df['lug_boot'] = df['lug_boot'].replace({'small':0, 'med':1, 'big':2})


# In[93]:


df['safety'].unique()
df['safety'] = df['safety'].replace({'low':0, 'med':1, 'high':2})


# In[94]:


# dataset post encoding
df.head()


# In[95]:


df.isna().sum()


# In[96]:


# Prepare input and output
dfi = df.iloc[:,:-1]
dfo=df['CAR']


# In[97]:


# Split the dataset in train and test
from sklearn.model_selection import train_test_split


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(dfi,dfo, test_size=0.30)
# Split the data in 7-:30 ratio where 70% train data will be used for training the algorithm and 30% test data will be used for 
# Prediction/Testing


# In[99]:


# Observe the shapes of train and test data: The 70:30 data split
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[100]:


# Train the data
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()


# In[101]:


dtc.fit(X_train, y_train)
dtc.score(X_train, y_train) # Showing 1 as the score which is considrered as ideal score, which is not found practically. We 
# should consider that data may be overfitting. Usually, we need to resolve the overfitting but here, for the sake of 
# implementation, we will continue.


# In[102]:


# Predict
y_pred = dtc.predict(X_test)


# In[103]:


# Lets form a dataframe of Observed vs Predicted Values
dff = pd.DataFrame({"Observed_Car_Values":y_test, "Predicted_Car_Values":y_pred})
dff.head()


# In[104]:


# Let us try to get the confusion matrix
from sklearn.metrics import confusion_matrix


# In[105]:


confusion_matrix(dff['Observed_Car_Values'], dff['Predicted_Car_Values'])


# In[106]:


# The Score is:
(112+23+360+15)/dff.shape[0] # Having 98.26 score is high score


# In[107]:


# Visualization: Lets import the required libraries
from sklearn.externals.six import StringIO # Initializes the data in stringIO format. Newline argument is like TextIO 
                                           #wrapper constructor
import pydotplus # It loads and parses data in DOT format, a grafical description language format
from sklearn import tree
import matplotlib.image as mpimage # To perform basic image loading, rescaling and display operations
import numpy as np


# In[108]:


# preprocessing
fname="d:/test.png" # any name 
dot_data = StringIO() #create object
out=tree.export_graphviz(dtc,
   feature_names=list(dfi.columns.values), 
   out_file=dot_data, 
    class_names= str(np.unique(dfo)), 
    filled=True,  
    special_characters=True,
    rotate=False)  


# In[109]:


grp=pydotplus.graph_from_dot_data(dot_data.getvalue()) #get graph data


# In[110]:


grp.write_png(fname)


# In[111]:


# PLot graph # May take time depending upon the soze of data
im=mpimage.imread(fname)
plt.figure(figsize=(100,200))
plt.imshow(im)
plt.show()

