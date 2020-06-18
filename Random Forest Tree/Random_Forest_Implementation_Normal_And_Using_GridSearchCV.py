#!/usr/bin/env python
# coding: utf-8

# In this code, I have implemented a Random Fores Algorithm on a popular datset named CAR DATASET. I will implement it with
# splitting the data using train_test_split 70:30 strategy. Post implementation, I will visualise the Random Forest.

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
df['buying'].unique()
df['buying'] = df['buying'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})

df['maint'].unique()
df['maint'] = df['maint'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})

df['doors'].unique()
df['doors'] = df['doors'].replace({'2':0, '3':1, '4':2, '5more':3})

df['persons'].unique()
df['persons'] = df['persons'].replace({'2':0, '4':1, 'more':2})

df['lug_boot'].unique()
df['lug_boot'] = df['lug_boot'].replace({'small':0, 'med':1, 'big':2})

df['safety'].unique()
df['safety'] = df['safety'].replace({'low':0, 'med':1, 'high':2})

# dataset post encoding
df.head()

df.isna().sum() #Check for null values

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
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
rfc.score(X_train, y_train) # Showing 99.91 as the score which is quite high. We should consider that data may be overfitting.
# Usually, we need to resolve the overfitting but here, for the sake of implementation, we will continue.

# Predict
y_pred = rfc.predict(X_test)

# Lets form a dataframe of Observed vs Predicted Values
dff = pd.DataFrame({"Observed_Car_Values":y_test, "Predicted_Car_Values":y_pred})
dff.head()

# Let us try to get the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(dff['Observed_Car_Values'], dff['Predicted_Car_Values'])

# The Score is:
(108+17+359+16)/dff.shape[0] # Having 96.33 score is a high score

# Let us implement Random Forest with GridSearchCV
# GridSearchCV: For an exhaustive search, GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, 
# “predict_proba”, “decision_function”, “transform” etc.
from sklearn.model_selection import GridSearchCV

# Create a param_grid : Its a dict or list of dictionaries. Dictionary with parameters names (string) as keys and lists of
# parameter settings to try as values, or a list of such dictionaries. This enables searching over any sequence of parameter
# settings.
lsd = {'n_estimators':[ a for a in range(1,100)],
    'criterion':['gini','entropy']}
# Gini and Entropy are the selection criterion in Trees. 

gs = GridSearchCV(rfc, param_grid=lsd, n_jobs=-1)

# Fit data
gs.fit(X_train, y_train)

# Estimate for the best score: Estimator which was chisen by the search, the best score and least loss
gs.best_estimator_

dff['Pred_cv']=gs.predict(X_test)
dff.head()

# Check the score
dff.groupby(['Observed_Car_Values','Pred_cv'])['Observed_Car_Values'].count()

print("Score: ", (108+21+362+18)/dff.shape[0]) # 98.07, a high score
