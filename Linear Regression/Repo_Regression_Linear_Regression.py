#!/usr/bin/env python
# coding: utf-8

# here I have tried to show the implementation of Linear Regression in the simplest way to understand its working.

import pandas as pd
import matplotlib.pyplot as plt #for visualization

# Load the dataset: 
df = pd.read_excel('d:/RegressionBookLM.xlsx')
df.head()

df.shape # 20 records, 3 Columns

# Here, on the basis of Expenditure and Cost variables, we will predict the sales 
# First, we will build single input regression model
# Then, we will build a multi input regression model

# Single Input Regression Model: Preprocessing
# Make the input and output variables
dfi = df[['Expenditure']] # Input: always has to me a multivalued arrangement hence [[..]]: Independent Variable
dfo= df['Sales'] # Output/Dependent Variable

# Get the Linear regression object
from sklearn.linear_model import LinearRegression

lr = LinearRegression() # Created an object of linearRegression

# Fit/Train the model with data
lr.fit(dfi,dfo)
# get the score
sc = lr.score(dfi,dfo)
print("Score is :", sc)

# Let us get the linear regression equation y= mx+c where m is the slope and c is constant: the intercept
# Get the coeff/slope
lr.coef_

# Get the intercept
lr.intercept_

# The equation
print("The Linear Regression  equation is: ")
print("Sales = (7.42858611*Expenditure Value) + 21.879367036580362")

# Predict The output
df['Sales_pred'] = lr.predict(dfi)
df.head(2)
# Here, we have observed sales and predicted sales

# Let us plot the observed and predicted sales 
plt.scatter(df['Expenditure'], df['Sales'], label='Observed') # Sales
plt.plot(df['Expenditure'], df['Sales_pred'], color='r', label='Predicted') # Predcted Sales/ Regression plot/Expected Sales
plt.title("Expenditure VS Sales")
plt.legend()
plt.show()

# Let us predict a value by out derived equation and using linera regression object
print("Regression Equation: Sales = (7.42858611*Expenditure Value) + 21.879367036580362")
exp = float(input("Enter Expenditure"))
print("Sales using equation = ", (7.42858611*exp) + 21.879367036580362)

exp = float(input("Enter Expenditure"))
print("Sales using Regression Object = ", lr.predict([[exp]]))

# We see that both the values are same with 7 decimal places
# Let us plot them on the regression plot
plt.scatter(df['Expenditure'], df['Sales'], label='Observed') # Sales
plt.plot(df['Expenditure'], df['Sales_pred'], color='r', label='Predicted') # Predcted Sales/ Regression plot/Expected Sales
plt.scatter(15, lr.predict([[15]]), marker='*', color='k', s=100, label="Prediction for Expenditure =15")
plt.title("Expenditure VS Sales")
plt.legend()
plt.show()

# Let us build multi inpur model
dfi = df[['Expenditure', 'Cost']] # This time more than one variable as input
dfo=df['Sales']

lr = LinearRegression() 
lr.fit(dfi,dfo)
lr.score(dfi,dfo) 

# Let us derive the multi input regression equation
# Get the Slope
lr.coef_ # Two slopes for two input variables

# Get the intercept
lr.intercept_

# Predict Sales
df['Sales_Pred_Multi_Input'] = lr.predict(dfi)
df.head(2)

print("Equation type: y = coeff1*x1 + coeff2*x2 + intercept")
print("The Regression Equation is:")
print("Sales = (7.25169741*Expenditure Value) + (0.61317216*Cost Value) + 11.295594591515822")

# Let try to predict Sales for Expenditure=10.5 and Cost = 21
print("Sales for Expenditure=15 and Cost = 21:")
print((7.25169741*10.5) + (0.61317216*21) + 11.295594591515822)

# These files can be saved as pkl file and can be sent where the receiver, using the Joblib, can access the regression funcion

