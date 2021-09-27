#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[36]:


df=pd.read_csv('diabetes.csv')


# In[37]:


df.head()


# In[38]:


df.dtypes


# In[39]:


df.columns


# In[40]:


df.info


# In[41]:


for i in df.columns :
    print (i ,'has ',df[i].isnull().sum(),'null values')


# In[42]:


pd. set_option('display.max_rows', 500)
pd. set_option('display.max_columns', 500)


# In[43]:


df._get_numeric_data().columns


# In[44]:


#To get the numerical features
numeric_=df.loc[:,df.dtypes!=np.object]
numeric_.columns


# In[45]:


#To get the categorical features
categoric_=df.loc[:,df.dtypes==np.object]
categoric_.columns


# In[46]:


#Display number of rows, columns
df.shape


# In[47]:


# Return boolean Series denoting duplicate rows.
df.duplicated().sum()


# In[48]:


#display unique values in each features
for col in df:
    print(col,df[col].unique())


# In[49]:


#count plot of diabetic and non diabetic persons.
df.diabetes.value_counts().plot(kind='bar',color=['C0','C1'])


# In[50]:


#count plot of number of females vs  males
df.gender.value_counts().plot(kind='bar',color=['C0','C1'])


# In[51]:


X=df.drop(['diabetes'],axis=1)
Y=df['diabetes']


# In[52]:


# separate dataset into train and test
'''
If you don't specify the random_state in your code, then every time you run(execute) your code a new random value is generated and the train and test datasets would have different values each time.
However, if a fixed value is assigned like random_state = 42 then no matter how many times you execute your code the result would be the same .i.e, same values in train and test datasets.
'''

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# In[53]:


X_train.corr()


# In[54]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = X_train.corr()
sns.heatmap(cor,annot=True,center=0,cmap=plt.cm.CMRmap_r)
plt.show()


# In[55]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[56]:


corr_features = correlation(X_train, 0.7)
len(set(corr_features))


# In[57]:


corr_features


# In[58]:


X_train.drop(corr_features,axis=1)
X_test.drop(corr_features,axis=1)


# In[59]:


sns.pairplot(df, hue ='diabetes')
# to show
plt.show()


# In[60]:


#function to replace the ','with '.'
def replacee(s):
    i=str(s).find(',')
    if(i>0):
        return s[:i] + '.' + s[i+1:]
    else :
        return s 


# In[61]:


#change the type to float
df['chol_hdl_ratio']=df['chol_hdl_ratio'].apply(replacee).astype(float)
df['bmi']=df['bmi'].apply(replacee).astype(float)
df['waist_hip_ratio']=df['waist_hip_ratio'].apply(replacee).astype(float)


# In[62]:


#machine learning algorithms cannot understand categorical values,so  mapping  male and female to 0 and 1 respectively
df.gender=df.gender.map({'male':0,'female':1})
df.diabetes=df.diabetes.map({'No diabetes':0,'Diabetes':1})


# In[63]:


inp=df.drop(['patient_number','age', 'hip', 'waist','gender','diabetes'],axis=1)


# In[64]:


inp.head()


# In[65]:


inp.dtypes


# In[66]:


tar=df['diabetes']


# In[67]:


#train test split
X_train, X_test, y_train, y_test = train_test_split(
    inp,
    tar,
    test_size=0.2,
    random_state=0)

X_train.shape, X_test.shape


# In[70]:


#standard scaler.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[71]:


#logistic regession
model = LogisticRegression()
model.fit(X_train, y_train)
pickle.dump(model,open('model.pkl','wb'))


# In[72]:


y_pred = model.predict(X_test)


# In[73]:


X_test[:1]


# In[74]:


y_pred


# In[75]:


model.score(X_test,y_test)


# In[ ]:





# In[ ]:




