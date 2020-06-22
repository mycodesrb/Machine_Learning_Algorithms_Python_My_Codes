#!/usr/bin/env python
# coding: utf-8

# In[1]:


# SVM stands for Support Vector Machine. Under SVM we have Support vector Regression and Support Vector Classification both
# Here we will see SVM for regression data. It's implementation is just the same for classification. The difference being
# using Support Vector Classification object instead of regression.
# Here, no external dataset is reffered. I have generated random numbers to simplify the understandability.


# In[2]:


# mport required libraries/packages
import pandas as pd
import numpy as np


# In[6]:


# Generate Data: Here, 40 random numbers are generated
X= np.sort(np.random.rand(40,1),axis=0) #column wise sort
X = np.sort(5 * np.random.rand(40, 1), axis=0) # 5 is just a number to increase the magnitude, any number can be taken
y=np.sin(X).ravel() # Sin() to get sine of X values and Ravel() falttens it to 1D


# In[9]:


X[:5]


# In[8]:


y[:5]


# In[10]:


# Let us plot X vs y to see the sine curve
import matplotlib.pyplot as plt


# In[11]:


plt.scatter(X,y)
plt.legend()
plt.show()


# In[12]:


# Make data frame
df = pd.DataFrame(X.tolist()) #to convert array to list and then to DF
df['Y']=y
df.columns=['X','Y']
df.head()


# In[13]:


# Let's mplement SVR algorithm
from sklearn.svm import SVR


# In[14]:


# We have 3 different kernels in SVR. We need to select the best one according to the vaues
# Linear would plot like a straight line irrespective of the spread of data
# Poly would plot a curve with regularization with dispersed data
# RBF: Radial Based Function would plot with most of the dispersed data with maximum regularization hence it is more used.
# Also, we get the lease SSE in case of RBF as compared to other two kernel types
# Lets create objects for the above 3 kernels
svr_lin = SVR(kernel='linear') 
svr_poly=SVR(kernel='poly')
svr_rbf=SVR(kernel='rbf')


# In[15]:


# For linear kernel
svr_lin.fit(df[['X']],df['Y'])
svr_lin.score(df[['X']],df['Y'])


# In[16]:


# for Poly kernel
svr_poly.fit(df[['X']],df['Y'])
svr_poly.score(df[['X']],df['Y'])


# In[18]:


# for RBF kernel
svr_rbf.fit(df[['X']],df['Y'])
svr_rbf.score(df[['X']],df['Y']) # we see it gives the maximum score


# In[19]:


# Let us predict for all 3 kernels 
df['y_lin']=svr_lin.predict(df[['X']])
df['y_poly']=svr_poly.predict(df[['X']])
df['y_rbf']=svr_rbf.predict(df[['X']])
df.head(2)


# In[21]:


#Let's plot the the regression line for all three kernels
plt.scatter(df['X'],df['Y'], label="Observed")
plt.plot(df['X'],df['y_lin'],label="Linear")
plt.plot(df['X'],df['y_poly'],label="Poly")
plt.plot(df['X'],df['y_rbf'],label="rbf")
plt.legend()
plt.show()
# Here we see that rbf has the curve with maximum values covered in its plot. 


# In[22]:


#Lets check with SSE for kernels
df['SE_lin']=(df['y_lin']-df['Y'])**2
df['SE_poly']=(df['y_poly']-df['Y'])**2
df['SE_rabf']=(df['y_rbf']-df['Y'])**2
df.head(2)


# In[24]:


df.sum() # we see that the minimum squared error is with RBF


# In[26]:


# Some imp points
# along with kernel type, we can specify the C parameter, often termed as the regularization parameter, which specifies to SVM 
# optimizer how much to avois missclassifying the values
# Along with kernel, C we and specify gamma parameter which defines how far the influence of a singke training example reaches
# For high gamma value, only nearby values are considered but for a low gamma value. farther values are also considered.
# We can get different scores by changing the values of C and gamma and can plot the curves. Depending upon the need and best
# scores, the best suitable parameters can be set as final calculating arrangement.

