#!/usr/bin/env python
# coding: utf-8

# In this code, I have implemented Naive Bayes Algorithm on a popular dataset named CAR DATASET. I will implement it with
# splitting the data using train_test_split 70:30 strategy. 

# Load Info About the Dataset
f = open('D:/car.names')
for i in f:
    print(i)

# Load Basic Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Load Data
df = pd.read_csv('D:/car.data', header=None,
                names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','CAR'])
df.head()

df.shape # Total 1728 records

# Let us have a look on the output : Shows 4 classes viz unacceptable, acceptable, very good and good
df['CAR'].unique()

sb.countplot(df['CAR'])

# Shows about 70% cars are unacceptable, about 400 carrs are acceptable. Very few comes under Very Good and Good Category

df.head(2)

# To fit data in a model, input values needs to be numeric. Lets check each column and encode the values in numeric form,
# if needed
# We can encode the data with 3 Popular encoding ways. Though, i will show the examples of all 3, following 2 points should be 
# kept in mind while encoding so as to optimize the processing time.
# 1: If the categories are very less, go for replacing or encode by using "astype"
# 2: If the categories are large and replacing seems tedious, then go for LabelEncoder

# Encoding by "replace"
df['buying'] = df['buying'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})

# By using "astype": Convert the type of variable into "category", and then apply category codes to encode
df['maint_cat'] = df['maint'].astype('category')
df['maint'] = df['maint_cat'].cat.codes

# By Label Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['doors'] = le.fit_transform(df['doors'])

df.head(2) # We can see buying, maint and doors are encoded, likewise we will encode all categorical data into numeric data

df['persons'].unique()
df['persons'] = df['persons'].replace({'2':0, '4':1, 'more':2})

df['lug_boot'].unique()
df['lug_boot'] = df['lug_boot'].replace({'small':0, 'med':1, 'big':2})

df['safety'].unique()
df['safety'] = df['safety'].replace({'low':0, 'med':1, 'high':2})

# dataset post encoding
df.head(2)

df = df.drop(['maint_cat'], axis=1)
df.head(2)

# Prepare input and output
dfi = df.iloc[:,:-1]
dfo=df['CAR']

# Split the dataset in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dfi,dfo, test_size=0.30)
# Split the data in 7-:30 ratio where 70% train data will be used for training the algorithm and 30% test data will be used for 
# Prediction/Testing

# Observe the shapes of train and test data: The 70:30 data split
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Train the data
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(X_train, y_train)
nb.score(X_train, y_train) 

# Predict
y_pred = nb.predict(X_test)

# Lets form a dataframe of Observed vs Predicted Values
dff = pd.DataFrame({"Observed_Car_Values":y_test, "Predicted_Car_Values":y_pred})
dff.head()


# Let us try to get the confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(dff['Observed_Car_Values'], dff['Predicted_Car_Values'])

# The Score is:
(20+2+318+18)/dff.shape[0]

# Let us try to implement interact in the dataset: Just for fun
def pred(buying,maint,doors,persons,lug_boot,safety):
    print("Depending upon the inputs, the predicted CAR type is ", nb.predict([[buying,maint,doors,persons,lug_boot,safety]]))
    
from ipywidgets import interact

interact(pred, buying=df['buying'].unique(),maint=df['maint'].unique(),doors=df['doors'].unique(),persons=df['persons'].unique(),lug_boot=df['lug_boot'].unique(),safety=df['safety'].unique())
