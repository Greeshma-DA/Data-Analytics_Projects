#!/usr/bin/env python
# coding: utf-8

# FAKE NEWS DETECTION WITH PYTHON

# In[11]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[27]:


#Read the data

df=pd.read_csv(r'C:\Users\greec\OneDrive\Desktop\news.csv')


# In[28]:


df.head()


# In[21]:


df.shape


# In[22]:


df.isnull().sum()


# In[23]:


#Split the dataset

labels=df.label
labels.head()


# In[24]:


#Split the dataset

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[25]:


#Initialize a TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[26]:


#Fit and transform train set, transform test set

tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[30]:


#Initialize a PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# In[32]:


#Predict on the test set and calculate accuracy

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[33]:


#Build confusion matrix

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




