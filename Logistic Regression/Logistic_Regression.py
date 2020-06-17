#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In this code, I have implemented Logistic Regression and then compared it with Linear Regression to get the idea of 
# Sigmoid Function and to get an idea of difference between regression and classification.
# Here I have used one of the popular Diabetes Dataset. You can get the dataset online or else I have included it in the 
# folder here itself.


# In[2]:


# import basic libraries
import pandas as pd
import numpy as np


# In[3]:


# Load the dataset
df = pd.read_csv("d:/datasets_33873_44826_diabetes.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


# Check if the dataset has null values
df.isnull().sum() # Found no null values


# In[7]:


# Some basic info: For any dataset to be fit for a model, the values should be numeric. If not, we have to make them numeric
# by different encodings. Here, all values are numeric so we can preprocess them easily. We have 8 columns as the input 
# variables and Outcome as the output variable. It is not a continuous data but either 0 or 1 which represents two classes
# for 1 being the presence of diabetes and 0 being the absense of diabetes. In the case of Linear Regression, the output is 
# always a mix of numeric data not category or class hence we call this as a classification algorithm as the output classifies
# a person diabetic or non diabetic rather than giving some numeric value for each person.

# Preprocessing
dfi = df.iloc[:,:-1] # input
dfo = df['Outcome']


# In[8]:


# Create Logistic regression object
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[9]:


# Fit the data and get the score
lr.fit(dfi,dfo)
lr.score(dfi,dfo)


# In[10]:


# Predict Outcome
df['Predict_LoR'] = lr.predict(dfi)
df.head(2)


# In[11]:


# Cross Check the accuracy
df.groupby(['Outcome','Predict_LoR'])['Outcome'].count()


# In[12]:


# The correct predictions/Total records
(448+147)/df.shape[0]


# In[13]:


# The above score = The score we got earlier so cross check is successful


# In[14]:


# Why not Linear Regression? Lets do it and check
from sklearn.linear_model import LinearRegression
lin = LinearRegression()

dfi=df.iloc[:,:-2]
dfo=df['Outcome']

lin.fit(dfi,dfo)
lin.score(dfi,dfo)


# In[15]:


# As we see that the score is quite poor as compared to Logistic regression hence NO to linear Reression for this dataset
# Lets predict
df['Predict_Lin'] = lin.predict(dfi)
df.head(2)


# In[16]:


# Here we see that the prediction of Linear Regression is a column of different values rather that of a class of 0 or 1
# So let us try to convert the prediction into a categorized column. To convert it, we need a function which will convert
# the colum values in either 1 or 0 depending upon some threshold value. If column value is > threshold value, it converts
# value in 1 else 0. This functionality is achieved by using Sigmoid function which calculates the threshold by the formula 
# shown in the following code:

# The sigmoid Function
def sigmoid(predicted_value):
    threshold = 1/(1+np.exp(-predicted_value))
    if threshold>=0.5:
        return 1
    else:
        return 0


# In[17]:


# Let us apply the above function to the prediction column
df['Converted_Prediction'] = df['Predict_Lin'].apply(lambda x:sigmoid(x))
df.head(2)


# In[18]:


# Now we have converted df['Predict_Lin'] into equivalent categorical values into df['Converted_Prediction']
# Lets cross check score
df.groupby(['Outcome','Converted_Prediction'])['Converted_Prediction'].count()


# In[19]:


# Correct outcomes/total records
(52+266)/df.shape[0]


# In[20]:


# A bit better as compared to Linear Regression score but still a poor score as compared to the Logistic Regression Score
# So this is the way we implement Logistic Regression. 

# Let Us do the Logistic Regression with train and Test split approach where we split training data and testing data into
# a suitable proportion. We train with train data and predict with test data which ensures that the data for training is not
# overfitting instead we are fitting less percentage of data to train and rest data to test the predictions/results. 

# Implementation is as as fllows:
df = pd.read_csv("d:/datasets_33873_44826_diabetes.csv")
df.head(2)


# In[21]:


# Preprocessing
dfi = df.iloc[:,:-1]
dfo=df['Outcome']


# In[22]:


# Import required library
from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(dfi,dfo, test_size=0.30) # 0.30 means divide data in 70:30 ratio, 0.30 is 
                                                                            # for 30% as test data

# Observe the size of training data: We will train the model with this data, X_train as input and y_train as output
print(X_train.shape, y_train.shape)

# Observe the size of test data: We will test/predict the model with this data, X_test as input and y_test as output
print(X_test.shape, y_test.shape)


# In[24]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)


# In[25]:


# Predict
y_prediction = lr.predict(X_test)

dff = pd.DataFrame({'Old_Outcome': y_test, 'Predicted_Outcome':y_prediction})
dff.head()


# In[26]:


# Crosscheck the accuracy
dff.groupby(['Old_Outcome','Predicted_Outcome'])['Predicted_Outcome'].count()


# In[27]:


# Correct Outcomes/total
(133+42)/dff.shape[0]


# In[28]:


# Just the same with little decimal difference
# So, this was the way for train and test split approach for Logistic Regression implementation


# In[ ]:




