#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[3]:


data=pd.read_csv('diabetic_data.csv')
data.head(100)


# In[ ]:





# In[4]:


data.columns


# In[5]:


data.isna().sum()


# In[6]:


data.shape


# In[7]:


data.diabetesMed 


# In[8]:


num_col=data._get_numeric_data().columns


# In[9]:


len(num_col)


# In[ ]:





# In[10]:


cat_col=data.select_dtypes(include=['object','category']).columns.tolist()


# In[11]:


len(cat_col)


# In[ ]:





# In[12]:


data.info


# In[13]:


len(data)


# In[14]:


len(data.columns)


# In[15]:


data.shape


# In[16]:


data.size


# In[17]:


from pandas_profiling import ProfileReport


# In[18]:


profile=ProfileReport(df=data,title='pandas profiling report',explorative=True)


# In[19]:


profile


# In[20]:


data.head()


# In[21]:


data.nunique()


# In[ ]:





# In[22]:


pd.get_option("display.max_columns")


# In[23]:


pd.set_option("display.max_columns", None)


# In[24]:


data.head()


# In[ ]:





# In[ ]:




