#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In this code, I have implemented K Means on driver dataset. It is quite a simple dataset for basic practice
# Dataset Info: Input Variables: Driver_ID, Distance_Feature, Speeding_Feature and Output Column: Pre_Class
# BUT the thing to remember is K Means is a Clustering algorithm which works on unSupervised Data which means that we do not
# have the Output/Goal Column. Here we cluster the available data or we group the data on certain required/available criterion.
# Let us see it's how to do it.


# In[2]:


# Load Basic Libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#load dataset
df=pd.read_csv("d:/driverdata.csv")
df.head(2)


# In[4]:


df.shape #4000 records


# In[5]:


# We will work on only input data, let's extract it (Also, Driver Id has no role so we can get rid of that)
dff = df.drop(['Driver_ID','Pre_Class'], axis=1)
dff.head(2) # This is the required dataset.


# In[6]:


dff.info()


# In[7]:


dff.isnull().sum()


# In[8]:


dff.head(2)


# In[10]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=2) # We need 2 clusters so


# In[12]:


# Input data
dfi = dff 

#Fit data
km.fit(dfi)


# In[13]:


# Predict
dff['Prediction'] = km.predict(dfi)
dff.head(2)


# In[15]:


dff['Prediction'].unique() # As we have chosen 2 clusters, predction will have 2 categories


# In[17]:


# Values in each Clusters
dff['Prediction'].value_counts()


# In[18]:


# Lets visualize the clusters
for a in dff['Prediction'].unique():
    dfc = dff[dff['Prediction']==a]
    plt.scatter(dfc['Distance_Feature'], dfc['Speeding_Feature'], label=a)
plt.title("Distance vs Speeding Clusters")    
plt.xlabel("Distance")
plt.ylabel("Speeding")
plt.legend()
plt.show()
# Here, we can clearly see the two clusters


# In[22]:


# Let us try to plot the same with 3 clusters
dfi = dff.iloc[:,[0,1]]
km = KMeans(n_clusters=3)
km.fit(dfi)
dff['Prediction_n3'] = km.predict(dfi)
for a in dff['Prediction_n3'].unique():
    dfc = dff[dff['Prediction_n3']==a]
    plt.scatter(dfc['Distance_Feature'], dfc['Speeding_Feature'], label=a)
plt.title("Distance vs Speeding Clusters")    
plt.xlabel("Distance")
plt.ylabel("Speeding")
plt.legend()
plt.show()
# We see 3 distince clusters


# In[23]:


# From the above 2 visualizations, we observe that the data values are plced in the respective groups based on some feature 
# similarity. We say, the distance of each data value from the center of thr group have some similarity among the cluster.
# That means each group must have one center point, called as centroid.
# Let us plot the centroids of each cluster


# In[32]:


dfi = dff.iloc[:,[0,1]]
km = KMeans(n_clusters=3)
km.fit(dfi)
#print the centroids
print(km.cluster_centers_)
dff['Pred'] = km.predict(dfi)
for a in dff['Pred'].unique():
    dfc = dff[dff['Pred']==a]
    plt.scatter(dfc['Distance_Feature'],dfc['Speeding_Feature'], label=a)
    #plot the centroid
    plt.scatter(km.cluster_centers_[a][0], km.cluster_centers_[a][1], marker='*', color='k',s=100)
plt.title("Distance vs Speeding Clusters and Centroids")    
plt.xlabel("Distance")
plt.ylabel("Speeding")
plt.legend()
plt.show()


# In[ ]:


# And now when we have brought the output clumn for the dataset by clustering, we can further implement different algorithms 
# on the dataset in a suervised way.

