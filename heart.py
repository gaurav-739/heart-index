#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


df=pd.read_csv(r"D:\Documents\heart\heart.csv")


# In[27]:


X=df.iloc[:,:13].values
Y=df.iloc[:,-1].values


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[29]:


regressor = GaussianNB() 
regressor.fit(X_train,Y_train)


# In[30]:


Y_pred = regressor.predict(X_test)


# In[31]:


pickle.dump(regressor, open('heart.pkl','wb'))

model = pickle.load(open('heart.pkl','rb'))


# In[32]:


print ("Accuracy : ", accuracy_score(Y_test, Y_pred)) 


# In[ ]:





# In[ ]:




