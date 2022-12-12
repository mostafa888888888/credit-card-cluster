#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[62]:


data=pd.read_csv("D:\\credit card.csv")
data.head()


# In[63]:


data.drop(['CUST_ID'],axis=1,inplace=True)


# In[64]:


data.info()


# In[65]:


data.isnull().sum()


# In[66]:


data['MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].replace(np.NaN,data['MINIMUM_PAYMENTS'].mean())
data['CREDIT_LIMIT']=data['CREDIT_LIMIT'].replace(np.NaN,data['CREDIT_LIMIT'].mean())


# In[67]:


data.isnull().sum()


# In[68]:


data.describe()


# In[69]:


data.hist(bins=44,figsize=(24,18))
plt.show()


# In[70]:


plt.figure(figsize=(24,18))
corr = data.corr()
sns.heatmap(corr,annot=True)
plt.show()


# In[71]:


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
x = min_max_scaler.fit_transform(data)
x[: 5]


# In[72]:


from sklearn.cluster import KMeans
num = []
for i in range(1, 11):
    model = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    model.fit(x)
    num.append(model.inertia_)


# In[73]:


plt.plot(range(1, 11), num)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# In[84]:


model = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_model = model.fit_predict(x)


# In[85]:


data['cluster']=y_model
data


# In[86]:


data['cluster'].value_counts()


# In[87]:


data['cluster'].hist()


# In[88]:


data.replace({'cluster':{0:"A",1:'B',2:"C",3:'D',4:'G'}},inplace=True)
data


# In[99]:


plt.scatter(x[y_model==0,0],x[y_model==0,1],s=100,c='red',label='cluser1')
plt.scatter(x[y_model==1,0],x[y_model==1,1],s=100,c='blue',label='cluser1')
plt.scatter(x[y_model==2,0],x[y_model==2,1],s=100,c='green',label='cluser1')
plt.scatter(x[y_model==3,0],x[y_model==3,1],s=100,c='cyan',label='cluser1')
plt.scatter(x[y_model==4,0],x[y_model==4,1],s=100,c='black',label='cluser1')


# In[ ]:




