#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.decomposition import PCA


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


na_dict = {'LotFrontage':['NA','#NA',''],'MasVnrArea':'NA',
           'GarageYrBlt':'NA','Electrical':'NA'}


# In[4]:


df_train = pd.read_csv('train.csv', 
                       keep_default_na=False,
                       na_values=na_dict)
df_test = pd.read_csv('test.csv', 
                       keep_default_na=False,
                       na_values=na_dict)


# In[5]:


df_train.shape


# In[6]:


df_train.info()


# # Outlier Detection

# In[7]:


df_train_out=df_train[df_train['SalePrice']<340000] 


# In[8]:


df_train_out.shape


# In[9]:


columns_numeric = list(df_train_out.select_dtypes(exclude='object').columns)
columns_numeric.remove('SalePrice')
columns_numeric.remove('Id')
print(columns_numeric)


# In[13]:


columns_object = list(df_train_out.select_dtypes(include='object').columns)
print(columns_object)


# # Preparetion of Validation Data

# In[14]:


X = df_train_out.drop('SalePrice',axis=1)
y = df_train_out['SalePrice']


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=13)


# # Handling of Missing Data

# In[16]:


from sklearn.impute import SimpleImputer


# In[17]:


imputer_numeric = SimpleImputer(strategy='median')
imputer_object = SimpleImputer(strategy='most_frequent')


# In[18]:


imputer_numeric.fit(X_train[columns_numeric])
imputer_object.fit(X_train[columns_object])


# In[19]:


X_train[columns_numeric] = imputer_numeric.transform(X_train[columns_numeric])
X_train[columns_object] = imputer_object.transform(X_train[columns_object])


# In[20]:


X_train[columns_numeric].isna().sum()


# In[21]:


X_train[columns_object].isna().sum()


# # Scaling of Numeric Data

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scalar=StandardScaler()
scalar.fit(X_train[columns_numeric])
X_train[columns_numeric]=scalar.transform(X_train[columns_numeric])


# In[24]:


X_train[columns_numeric]


# # Encoding by OneHotEncoding

# In[25]:


encoder=OneHotEncoder(handle_unknown='ignore', sparse=False)
encoder.fit(X_train[columns_object])
temp=encoder.transform(X_train[columns_object])


# In[26]:


temp_df=pd.DataFrame(temp)
columns_encoded = list(temp_df.columns)
X_train = pd.concat([X_train.reset_index(),temp_df.reset_index()], axis=1)


# In[27]:


temp_df.shape


# In[28]:


temp_df.head()


# In[29]:


X_train.shape


# In[30]:


df_corr=df_train_out.corr()['SalePrice']


# In[31]:


X_corr=df_corr[df_corr.apply(lambda x:x>=-0.1 and x<=0.25)]


# # Model Building

# In[40]:


params={'n_estimators':[10,20,40,60,100],
       'min_samples_leaf':[2,3,4,5,6],
        'max_depth':[2,3,4,5,6],
        'min_samples_split':[3,4,5,6]}


# In[41]:


from sklearn.model_selection import GridSearchCV


# In[42]:


grid_cv=GridSearchCV(RandomForestRegressor(),params,n_jobs=-1)


# In[43]:


grid_cv.fit(X_train[columns_encoded+columns_numeric],y_train)


# In[44]:


grid_cv.best_estimator_


# In[45]:


model=RandomForestRegressor(max_depth=6, min_samples_leaf=2, min_samples_split=4,
                      n_estimators=40,random_state=13)
model.fit(X_train[columns_encoded+columns_numeric],y_train)


# # Train Accuracy

# In[46]:


model.score(X_train[columns_encoded+columns_numeric],y_train)


# # Validation Accuracy

# In[47]:


X_val[columns_numeric] = imputer_numeric.transform(X_val[columns_numeric])
X_val[columns_object] = imputer_object.transform(X_val[columns_object])


# In[48]:


X_val[columns_numeric] = scalar.transform(X_val[columns_numeric])


# In[49]:


temp=encoder.transform(X_val[columns_object])
temp_df=pd.DataFrame(temp)
X_val=pd.concat([X_val.reset_index(), temp_df.reset_index()], axis=1)


# In[50]:


model.score(X_val[columns_encoded+columns_numeric], y_val)


# # Testing Accuracy

# In[51]:


na_dict1={'LotFrontage':'NA','Utilities':'NA','Exterior1st':'NA',
         'Exterior2nd':'NA','MasVnrArea':'NA', 'BsmtFinSF1':'NA',
         'BsmtFinSF2':'NA','BsmtUnfSF':'NA','TotalBsmtSF':'NA','BsmtFullBath':'NA',
          'BsmtHalfBath':'NA','KitchenQual':'NA','Functional':'NA','GarageYrBlt':'NA',
          'GarageCars':'NA','GarageArea':'NA', 'SaleType':'NA'}
df_test = pd.read_csv('test.csv', 
                       keep_default_na=False,
                       na_values=na_dict1)


# In[52]:


df_test.isna().sum()


# In[53]:


df_test[columns_numeric] = imputer_numeric.transform(df_test[columns_numeric])
df_test[columns_object] = imputer_object.transform(df_test[columns_object])


# In[54]:


df_test[columns_numeric] = scalar.transform(df_test[columns_numeric])


# In[55]:


temp=encoder.transform(df_test[columns_object])
temp_df=pd.DataFrame(temp)
df_test=pd.concat([df_test.reset_index(), temp_df.reset_index()], axis=1)


# In[56]:


y_pre=model.predict(df_test[columns_encoded+columns_numeric])


# In[57]:


y_pre


# In[58]:


df_submission=pd.DataFrame({'Id':df_test['Id'],'SalePrice':y_pre})


# In[59]:


df_submission


# In[60]:


df_submission.to_csv('sub.csv',index=False)


# In[ ]:




