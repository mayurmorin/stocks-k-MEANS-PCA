
# coding: utf-8

# ## Problem 1: There are various stocks for which we have collected a data set, which all stocks are apparently similar in performance

# <p> Dataset Link https://drive.google.com/file/d/1pP0Rr83ri0voscgr95-YnVCBv6BYV22w/view </p>

# ### Importing Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime


# ### Loading Data

# In[2]:


#Read CSV (comma-separated) file into DataFrame
data = pd.read_csv('data_stocks.csv')


# ### Data Exploration

# In[3]:


data.head()  #Returns the first 5 rows of data dataframe


# In[4]:


data.describe() #The summary statistics of the data dataframe


# In[5]:


data.info() #Prints information about data DataFrame.


# In[6]:


data.columns #Columns of data dataframe


# In[7]:


data.shape #Return a tuple representing the dimensionality of data DataFrame.


# In[8]:


data.isnull().values.any() #Check for any NA’s in the dataframe.


# ### Data Visualization

# #### Dropping DATE and SP500 columns ( For PCA and KMeans clustering )

# In[9]:


data_new = data.copy() #Making a copy of the data dataframe


# In[10]:


data_new.drop(['DATE', 'SP500'], axis=1, inplace = True) #Removing the Date and SP500 columns


# In[11]:


data_new.head() #Returns the first 5 rows of data dataframe


# In[12]:


data_new.shape #Return a tuple representing the dimensionality of data DataFrame.


# In[13]:


data_new.columns #Return columns of data dataframe.


# ### PCA

# In[14]:


#Creating an instance of PCA
pca = PCA(n_components=3)


# In[15]:


#Fitting the pca object
pca.fit(data_new)


# In[16]:


#Transforming the data_new dataframe
data_new_reduced = pca.transform(data_new)


# In[17]:


data_new_reduced.shape #Return a tuple representing the dimensionality of data_new_reduced DataFrame.


# In[18]:


data_new_reduced[:1].shape #Return a tuple representing the dimensionality of data_new_reduced dataframe's 1st row.


# In[19]:


#Scatter Plot
plt.figure(figsize=(16,9))
plt.scatter(data_new_reduced[:,0],data_new_reduced[:,1])
plt.ylabel('PC1')
plt.xlabel('PC2')
plt.show()


# ### KMeans Clustering

# In[20]:


pca.explained_variance_ #Returns explained variance array


# In[21]:


data_new_reduced #Transformed data_new dataframe


# In[22]:


#Finding k and intertia for KMeans clustering using elbow method
k = []
inertia = []
for i in range(1,20):
    k_means = KMeans(n_clusters = i)
    k_means.fit(data_new_reduced)
    k.append(i)
    inertia.append(k_means.inertia_)


# In[23]:


inertia #Inertia List Data


# In[24]:


#Plot to find number of clusters (Elbow Method)
plt.figure(figsize=(16,9))
plt.plot(k,inertia)
plt.show()


# In[25]:


#Initializing and fitting KMeans
km = KMeans(n_clusters = 5)
km.fit(data_new)


# In[26]:


#Predicted values using KMeans
y_predict = km.predict(data_new)


# In[27]:


#Scatter Plot
x = data_new_reduced[:,0]
y = data_new_reduced[:,1]
plt.figure(figsize=(16,9))
plt.scatter(x, y, c = y_predict, alpha=0.5)
plt.show()


# In[28]:


#Adding 'Y_PREDICT' column in data_new dataframe
data_new['Y_PREDICT'] = y_predict


# In[29]:


data_new.head() #Returns the first 5 rows of data_new dataframe


# In[30]:


#Returns 'Y_PREDICT' column containing counts of unique values in data_new dataframe.
data_new['Y_PREDICT'].value_counts()


# ### Problem 1 

# In[31]:


#Read CSV (comma-separated) file into DataFrame
stocks= pd.read_csv('data_stocks.csv')


# In[32]:


stocks.head() #Returns the first 5 rows of stocks dataframe


# In[33]:


#Adding a new column 'NEW_DATE' in stocks dataframe 
stocks['NEW_DATE'] = pd.to_datetime(stocks['DATE'],unit='s')


# In[34]:


cols = stocks.columns.tolist() #Creating a list of columns from stocks dataframe


# In[35]:


cols = cols[-1:] + cols[:-1] #Making 'NEW_DATE' as first column


# In[36]:


cols #cols list data


# In[37]:


#Removing 'DATE' and 'SP500' columns
cols.remove('DATE')
cols.remove('SP500')


# In[38]:


cols #cols list data


# In[39]:


stocks.drop(columns=['DATE','SP500'],axis=1,inplace=True) #Removing 'DATE' and 'SP500' from stocks dataframe


# In[40]:


stocks.head() #Returns the first 5 rows of stocks dataframe


# In[41]:


df = stocks[cols] #Creating a df objects which has cols list column data from stocks dataframe


# In[42]:


df.head() #Returns the first 5 rows of df dataframe


# In[43]:


df.shape #Return a tuple representing the dimensionality of df DataFrame.


# In[44]:


#Setting NEW_DATE as index 
df.set_index('NEW_DATE',inplace=True)


# In[45]:


df.head()  #Returns the first 5 rows of df dataframe


# In[46]:


df_transpose = df.transpose() #Creating transpose of the df dataframe


# In[47]:


df_transpose.head() #Returns the first 5 rows of df_transpose dataframe


# In[48]:


#Creating an instance of PCA
pca_new = PCA(n_components=3)


# In[49]:


#Fitting and tranforming the df_transpose dataframe
df_transpose_reduced = pca_new.fit_transform(df_transpose)


# In[50]:


df_transpose_reduced.shape  #Return a tuple representing the dimensionality of df_transpose_reduced DataFrame.


# In[51]:


#Scatter Plot
plt.figure(figsize=(16,9))
plt.scatter(df_transpose_reduced[:,0],df_transpose_reduced[:,1])
plt.ylabel('PC1')
plt.xlabel('PC2')
plt.show()


# In[52]:


#Returns explained variance array
pca_new.explained_variance_


# In[53]:


#Finding k_new and intertia_new list data for KMeans clustering using elbow method
k_new = []
inertia_new = []
for i in range(2,10):
    km_new=KMeans(n_clusters=i)
    km_new.fit(df_transpose)
    k_new.append(i)
    inertia_new.append(km_new.inertia_)


# In[54]:


inertia_new #inertia_new list data


# In[55]:


#Plot to find number of clusters (Elbow Method)
plt.figure(figsize=(16,9))
plt.plot(k_new,inertia_new)
plt.show()


# In[56]:


#Initializing and fitting KMeans
km = KMeans(n_clusters = 6)
km.fit(df_transpose)


# In[57]:


#Predicted values using KMeans
y_predict_new = km.predict(df_transpose)


# In[58]:


#Scatter Plot
plt.figure(figsize=(16,9))
plt.scatter(df_transpose_reduced[:,0],df_transpose_reduced[:,1],c=y_predict_new)
plt.show()


# In[59]:


#Adding 'y_predict_new' values in df_transpose dataframe creating 'Y_PREDICT' column
df_transpose['Y_PREDICT'] = y_predict_new


# In[60]:


#Returns 'Y_PREDICT' column containing counts of unique values in df_transpose dataframe.
df_transpose['Y_PREDICT'].value_counts()


# In[61]:


df_transpose.head() #Returns the first 5 rows of df_transpose dataframe


# In[62]:


#Apparently similar performing stocks of Type-1 are following:
df_transpose.loc[df_transpose['Y_PREDICT']==0]


# In[63]:


#Apparently similar performing stocks of Type-2 are following:
df_transpose.loc[df_transpose['Y_PREDICT']==1]


# In[64]:


#Apparently similar performing stocks of Type-3 are following:
df_transpose.loc[df_transpose['Y_PREDICT']==2]


# In[65]:


#Apparently similar performing stocks of Type-4 are following:
df_transpose.loc[df_transpose['Y_PREDICT']==3]


# In[66]:


#Apparently similar performing stocks of Type-5 are following:
df_transpose.loc[df_transpose['Y_PREDICT']==4]


# In[67]:


#Apparently similar performing stocks of Type-6 are following:
df_transpose.loc[df_transpose['Y_PREDICT']==5]


# ## Problem 2:
# ### How many Unique patterns that exist in the historical stock data set, based on fluctuations in price.

# In[68]:


#There are 5 unique patterns that exists in historical stock data set, based on fluctuation in price ( Observed Using KMeans Clustering Elbow Method )
#Pattern-1 stocks (based on fluctuations in price):
data_new.loc[data_new['Y_PREDICT']==0]


# In[69]:


#Pattern-2 stocks (based on fluctuations in price):
data_new.loc[data_new['Y_PREDICT']==1]


# In[70]:


#Pattern-3 stocks (based on fluctuations in price):
data_new.loc[data_new['Y_PREDICT']==2]


# In[71]:


#Pattern-4 stocks (based on fluctuations in price):
data_new.loc[data_new['Y_PREDICT']==3]


# In[72]:


#Pattern-5 stocks (based on fluctuations in price):
data_new.loc[data_new['Y_PREDICT']==4]


# # Problem 3:

# ### Identify which all stocks are moving together and which all stocks are different from each other.

# In[73]:


df_new = pd.read_csv('data_stocks.csv') #Read CSV (comma-separated) file into DataFrame


# In[74]:


df_new.head() #Returns the first 5 rows of df_new dataframe


# In[75]:


df_new.shape #Returns a tuple representing the dimensionality of df_new dataframe.


# In[76]:


#Removing 'DATE' and 'SP500' columns from df_new dataframe
df_new.drop(columns=['DATE','SP500'],inplace=True,axis=1)


# In[77]:


#Listing all the df_new dataframe columns
category_cols = df_new.columns


# In[78]:


#Creating the columns with the difference of the previous row 
for cat in category_cols:
    df_new["DIFF_"+ cat] = df_new[cat] - df_new[cat].shift(periods=1)


# In[79]:


df_new.shape #Returns a tuple representing the dimensionality of df_new dataframe.


# In[80]:


df_new.drop(category_cols,axis=1,inplace=True) #Removing the category_cols list columns from df_new dataframe


# In[81]:


df_new.shape #Returns a tuple representing the dimensionality of df_new dataframe.


# In[82]:


df_new.head() #Returns the first 5 rows of df_new dataframe


# In[83]:


#Removing the rows which containd NaN
df_new.dropna(inplace=True)


# In[84]:


df_new.head() #Returns the first 5 rows of df_new dataframe


# In[85]:


df_new_corr = df_new.corr() #Computes pairwise correlation of columns of df_new dataframe


# In[86]:


df_new_corr #Pairwise correlation dataframe of columns of df_new dataframe

