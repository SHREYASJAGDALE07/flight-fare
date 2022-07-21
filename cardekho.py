#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv('car data.csv')


# In[5]:


df.head()


# In[7]:


df.shape


# In[12]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[13]:


#missing and null value
df.isnull().sum()


# In[14]:


df.describe()


# In[15]:


df.columns


# In[21]:


final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[22]:


final_dataset.head()


# In[23]:


final_dataset['Current_Year']=2020


# In[24]:


final_dataset.head()


# In[25]:


final_dataset['no_years']=final_dataset['Current_Year']-final_dataset['Year']


# In[26]:


final_dataset.head()


# In[27]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[28]:


final_dataset.head()


# In[29]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[30]:


final_dataset.head()


# In[35]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[36]:


final_dataset.head()


# In[37]:


final_dataset.corr()


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


sns.pairplot(final_dataset)


# In[40]:


sns.heatmap(final_dataset)


# In[44]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[ ]:





# In[45]:


#independent and dependent featuures
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[46]:


X.head()


# In[48]:


y.head()


# In[49]:


##feature important
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[50]:


print(model.feature_importances_)


# In[54]:


#plot graph of featuire importance for bvetter visualization
feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[59]:


X_train.shape


# In[62]:


from sklearn.ensemble import RandomForestRegressor
import numpy as np
rf_random=RandomForestRegressor()


# In[63]:


###hyper-parameters
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
print(n_estimators)


# In[64]:


from sklearn.model_selection import RandomizedSearchCV


# In[65]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[67]:


#Create the random grid
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[68]:


rf=RandomForestRegressor()


# In[76]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter=10, cv = 5, verbose=2, random_state=42,n_jobs=1)


# In[77]:


rf_random.fit(X_train,y_train)


# In[79]:


prediction=rf_random.predict(X_test)


# In[80]:


prediction


# In[83]:



sns.distplot(y_test-prediction)


# In[84]:


plt.scatter(y_test,prediction)


# In[85]:


from sklearn import metrics


# In[86]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[87]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




