#!/usr/bin/env python
# coding: utf-8

# Apriori algorithm is a very much used algorithm. Majorly for recommending products along with the main products.
# Here, we will see how a dataset is converted into an improvised dataset which will be as good as a Recommendation Engine.
# I have used a popular dataset named BreadBasket. It has 4 columns viz. Date (Date of Purchase), Time(Time of Purchase),
# Transaction(Number of Purchase made in the store at that date and time) and Item (What is purchased). This dataset is an
# ideal dataset for learning Ariori Algorithm.

# Load the required Packages
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("d:/BreadBasket_DMS.csv")
df.head(2)

df.shape # Over 21k records

# Let us understand the Dataset
df.info()

len(df['Date'].unique()) # All the data is of 159 days which means Transactions are repeated 

len(df['Time'].unique()) # It shows 8240 times means all the products were purchased in these many time instances, hence 
                         # time also shows Transactions are repeated

len(df['Transaction'].unique()) # 9531 which means total transaction are 9531 and total records are 21k+. This shows that
                                # transactions are repeated which may include the returns also 

len(df['Item'].unique()) # Total items are 95 which were purchased in different transactions at different ties on different 
                        # days: Items also repeates

# Data Cleansing: Mostly a data contains unwanted values in the form of NaN, None, NONE, '?', '-'. These are the most found
# unwanted values. Let us check:
#for NaN
df.isnull().sum() # No NaN values

df[(df['Item']=="NONE") | (df['Item']=="None") | (df['Item']=="?") | (df['Item']=="-")]

# We found that Item column has NONE values. Lets get rid of them
df1 = df.drop(df[df['Item']=='NONE'].index)
df1.shape

# Total 21293 - 20507 = 786 NONE values were removed. Just because NONE was in Item column, we could not replace it with 
# any other item.

# Basically we will need Transaction and Items only for the algorithm, so lets check in Transaction as well
df[(df['Transaction']=="NONE") | (df['Transaction']=="None") | (df['Transaction']=="?") | (df['Transaction']=="-")]
# Nothing found

# Now, let us see Categorized Transaction w.r.t items: We want to check whethere transaction is done or not, and item sold or
# not
df1.groupby(['Transaction','Item'])['Item'].count() 

# Now we have a datafram with transaction and sold items. We will do follwing things on this data Frame
# Unstack it so that we get NaN for items which are not sold in that transaction else a value
# We will reset the index
# We will replace NaN with 0 so that we get a 0 or 1 data
# We ill reindex the data wit Transaction as the indexing column
dt  = df1.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
dt

# We want data with just 0 or 1 but the above data has va,ues other than 0 or 1, for ex.
dt[dt['Coffee']>1]['Coffee'].head(2)

# We will replace anything > 1 with 1 and 0 with 0
def setzero_one(x):
    if x>=1:
        return 1
    else:
        return 0

dt = dt.applymap(setzero_one)
dt

# Cross Check
dt[dt['Coffee']>1]['Coffee'].head(2) # Nothing found

# Here, we will implement the Apriori Algorithm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules 

basket = apriori(dt,min_support=0.02, use_colnames=True)
basket.head()

# We sort it on the basis of support of the items
df_basket = basket.sort_values(by='support', ascending=False)
df_basket.head()

# Now let us apply association rules: We use lift as the metric
rules = association_rules(df_basket,metric='lift',min_threshold=1)
rules.head()

dt2 = rules.sort_values(by='lift', ascending=False)
dt2.head()

recm_eng = dt2.iloc[:,[0,1]]
recm_eng.head() # This is out recommendation Engine: For an antecedent, recommendation is consequent

# To recommend, we need to access the above dataset not in the usual way. 
# recm_eng[recm_end['antecedents']=='(cake)'] -this won't work here

# Here we use frozenset as shown below:
recm_eng[recm_eng['consequents']==frozenset({'Cake'})] # it works

# Recommendation Function
def recm(item):
    print("Recommended item(s) with "+ item + " are: ")
    print( recm_eng[recm_eng['antecedents']==frozenset({item})]['consequents'])

recm('Cake')
recm('Coffee')
# Bingo
