#!/usr/bin/env python
# coding: utf-8

# In[37]:


# In this code, I have implemented KNN. While Iplementing it, i have tried to visualise its values.
# Here I have used one of the popular Diabetes Dataset. You can get the dataset online or else I have included it in the 
# folder here itself.


# In[38]:


# About Dataset: Iris Dataset:
# Total 5 Columns, 2 for sepals's length and width and 2 for Petal's length and width along with a goal columns as class
# Depending on the sepal's/petal's dimension, the class of flower is decided.


# In[39]:


# import basic libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[40]:


# Load the dataset
df = pd.read_csv("d:/iris.data", header=None, names=['s_length','s_width','p_length','p_width','class'])
df.head()


# In[41]:


df.shape # 150 records


# In[42]:


df.info()


# In[43]:


# Check if the dataset has null values
df.isnull().sum() # Found no null values


# In[44]:


# Check the output categories
df['class'].unique() # 3 Categiries found


# In[45]:


# Let us try to visualize the dataset with different output categories
plt.scatter(df['s_length'],df['s_width'])
plt.title("Sepal Lenght vs Sepal")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
# Here we can not see any distinguished output classes


# In[46]:


# Let us distinguish the output classes
for a in df['class'].unique():
    dfc = df[df['class']==a]
    plt.scatter(dfc['s_length'],dfc['s_width'], label=a)
plt.title("Sepal Lenght vs Sepal")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
# Here we see the output classes in different colors


# In[47]:


# Some basic info: For any dataset to be fit for a model, the values should be numeric. In the above dataset, all values
# are numeric. Let us try with sepal's length and width.

# Preprocessing
dfi = df.iloc[:,[0,1]] # input
dfo = df['class']


# In[48]:


# Create Logistic regression object
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[49]:


# Fit the data and check the score
knn.fit(dfi,dfo)
knn.score(dfi,dfo) # 83.33, a good score


# In[50]:


# Predict Outcome
df['Pred_Class'] = knn.predict(dfi)
df.head(2)


# In[51]:


# Cross Check the accuracy
df.groupby(['class','Pred_Class'])['class'].count()


# In[52]:


# The correct predictions/Total records
(49+38+38)/df.shape[0] #Verified..!!


# In[66]:


# Let us visualize the missing values of output class. For this, we will compare observed and predicted vaues of output class
df.head(2)


# In[75]:


plt.figure(figsize=(7,5))
for a in df['class'].unique():
    dfc = df[df['class']==a]
    plt.scatter(dfc['s_length'], dfc['s_width'], label=a) # Observed Values
df_miss = df[df['class'] != df['Pred_Class']] # Values present in class but not in predicted class
print("There are " + str(df_miss.shape[0]) +" missing values found")
plt.scatter(df_miss['s_length'], df_miss['s_width'], marker="*", color='w', s=80, label="Missing Values")
plt.title('Sepal Length vs Sepal Width with Misiing Values')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()
# This can be visualized for different values of n_neighbors, if required


# In[76]:


# Now let us check the score for different values of k_neighbors: We usually select the value depending upon the best score
dfi = df.iloc[:,[0,1]]
dfo=df['class']
sc, ls_score, ls_nbrs=0,[],[]
for n in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=n) # This value is considered by the algorith to compare the nearest distance with
                                            # these many number of nearest nodes
    knn.fit(dfi,dfo)    
    sc = knn.score(dfi,dfo)
    ls_nbrs.append(n)
    ls_score.append(sc)
    print("Score for n_neighbors = "+str(n)+ " is:", sc)
print('_______________________________________________')
print('Maximum Score is = ', max(ls_score), 'with n_neighbors =',ls_nbrs[ls_score.index(max(ls_score))])


# In[108]:


# Let us predict and visualize some manual values and check whether the predictions are right or wrong
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(dfi,dfo)

for a in df['class'].unique():
    dfc = df[df['class']==a]
    plt.scatter(dfc['s_length'],dfc['s_width'], label=a)
plt.scatter(6.1, 3.3, marker='v', color='k',label=knn.predict([[6.1,3.3]])) # Predicting manual values
plt.title("Sepal Lenght vs Sepal")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


# In[103]:


# Verifying the above prediction: to verify, we need to get the shortest distances of nearest nodes w.r.t our manual values
# Function for getting distance
import math

def getdist(x1,y1, x2,y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


# In[104]:


# Get the distance between our manual values and all the values in data frame and create a new distance column
for d in range(df.shape[0]):
    df['dist'][d] = getdist(6.1, 3.3, df['s_length'][d], df['s_width'][d])
df.head()    


# In[112]:


# Get the shortest distance and read its predicted class
df.sort_values(by='dist').head(1)['Pred_Class'].values[0]


# In[ ]:


# The predicted class is verified for n_neighbors=9 but not all the times we get accurate result unless we repeatedly test
# the scores by adjusting the n_neighbors value and then only predict the values. By this way, we can implement KNN algorithm 
# by same process for petal length and width or by considering all 4 input variables.

