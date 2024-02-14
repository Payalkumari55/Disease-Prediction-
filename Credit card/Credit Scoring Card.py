#!/usr/bin/env python
# coding: utf-8

# # Credit Scoring Model

# In[54]:


# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sns.set(style='white', context='notebook', palette='deep')
pd.options.display.max_columns = 100


# # Exploratory Data Analysis

# # Data 1

# In[9]:


train = pd.read_csv("C:/Users/ranji/Desktop/python/project/Code Alpha project/Credit card/Data/train.csv")


# In[10]:


train.head()


# In[12]:


train.shape


# In[13]:


train.info()


# In[14]:


train.describe()


# In[15]:


train.isnull().sum()


# # Data 2

# In[16]:


test = pd.read_csv("C:/Users/ranji/Desktop/python/project/Code Alpha project/Credit card/Data/test.csv")


# In[17]:


test.head()


# In[18]:


test.info()


# In[19]:


test.describe()


# In[20]:


test.isnull().sum()


# In[21]:


test.shape


# # Target Distribution

# In[23]:


ax = sns.countplot(x = train.jumlah_kartu ,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = 3874)
ax.set_xlabel('Financial difficulty in 2 years')
ax.set_ylabel('Frequency')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=4000)

plt.show()


# In[25]:


ax = sns.countplot(x = train.kode_cabang ,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = 3874)
ax.set_xlabel('Financial difficulty in 2 years')
ax.set_ylabel('Frequency')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=4000)

plt.show()


# In[27]:


ax = sns.countplot(x = train. skor_delikuensi,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = 3874)
ax.set_xlabel('Financial difficulty in 2 years')
ax.set_ylabel('Frequency')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=4000)

plt.show()


# # Detecting Outliers

# In[32]:


def detect_outliers(df,n,features):
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers

# detect outliers from Age, SibSp , Parch and Fare
# These are the numerical features present in the dataset
Outliers_to_drop = detect_outliers(train,2,["jumlah_kartu",                            
                                            "outstanding",                             
                                            "limit_kredit",                            
                                            "tagihan",                                 
                                            "total_pemakaian_tunai",                   
                                            "total_pemakaian_retail",                  
                                            "sisa_tagihan_tidak_terbayar",                                       
                                            "rasio_pembayaran",                        
                                            "persentasi_overlimit",                    
                                            "rasio_pembayaran_3bulan",                 
                                            "rasio_pembayaran_6bulan",                 
                                            "skor_delikuensi",                         
                                            "jumlah_tahun_sejak_pembukaan_kredit",     
                                            "total_pemakaian",                         
                                            "sisa_tagihan_per_jumlah_kartu",           
                                            "sisa_tagihan_per_limit",                  
                                            "total_pemakaian_per_limit",               
                                            "pemakaian_3bln_per_limit",                
                                            "pemakaian_6bln_per_limit",                
                                            "utilisasi_3bulan",                        
                                            "utilisasi_6bulan"])


# In[33]:


train.loc[Outliers_to_drop]


# In[34]:


train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# # Merging Datasets

# In[35]:


train_len = len(train)
dataset =  pd.concat(objs=[train,test], axis=0).reset_index(drop=True)


# In[36]:


dataset.shape


# # Exploring Variables

# In[40]:


# Correlation matrix
g = sns.heatmap(train.corr(),annot=False, fmt = ".2f", cmap = "coolwarm")


# # Exploring Tagihan

# In[42]:


dataset.tagihan.describe()


# In[43]:


dataset.skor_delikuensi.describe()


# In[44]:


dataset.tagihan = pd.qcut(dataset.tagihan.values, 5).codes


# In[47]:


g  = sns.catplot(x="tagihan",y="outstanding",data=dataset,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[67]:


g  = sns.catplot(x="sisa_tagihan_per_limit",y="skor_delikuensi",data=dataset,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Exploring 

# In[55]:


# Choose a regression model (Random Forest Regressor)
regressor = RandomForestRegressor(random_state=42)


# In[57]:


# Assuming 'X' contains your features and 'y' contains your target variable
X = dataset.drop(columns=['X'])  # Features
y = dataset['X']  # Target variable


# In[58]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[59]:


# Choose a regression model (Random Forest Regressor)
regressor = RandomForestRegressor(random_state=42)


# In[ ]:




