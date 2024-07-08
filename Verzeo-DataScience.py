#!/usr/bin/env python
# coding: utf-8

# ### Name: Roochita Ikkurthy
# ### Mail: roochita27@gmail.com

# ## Problem Statement: To perform classification analysis on Iris dataset. Perform any two classification algorithms and compare the accuracy.

# In[15]:


# importing pandas
import pandas as pd


# In[16]:


# reading data from the iris csv file
iris_df = pd.read_csv("Iris.csv")


# In[17]:


iris_df


# In[18]:


# defining data and label
x = iris_df.iloc[:, 1:5].values
y = iris_df.iloc[:, 5].values


# In[19]:


# splitting the data arrays in two subsets: training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ### Classification Algorithm : Decision Tree Classifier

# In[20]:


# importing DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier


# In[21]:


# tree object
dtc = DecisionTreeClassifier(random_state = 0)


# In[22]:


# train model
dtc.fit(x_train, y_train)


# In[23]:


y_dtc_pred = dtc.predict(x_test)


# In[24]:


y_dtc_pred


# In[25]:


# importing confusion_matrix and accuracy_score to evaluate the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score


# In[26]:


confusion_matrix(y_test, y_dtc_pred)


# In[27]:


#accuracy of the Decision Tree Classifier on Training data
accuracy_score(y_test, y_dtc_pred)


# In[28]:


# printing the accuracy in percentage
dtc_accuracy = accuracy_score(y_test, y_dtc_pred)*100
print('Accuracy of our model is equal to ' + str(round(dtc_accuracy, 2)) + '%')


# ### Classification Algorithm : Support Vector Machines(SVM)

# In[29]:


# importing StandardScaler
from sklearn.preprocessing import StandardScaler


# In[30]:


# SVM object
sc = StandardScaler()


# In[31]:


# scaling the training data
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[32]:


# Applying SVC (Support Vector Classification)
from sklearn.svm import SVC


# In[33]:


svc = SVC(kernel = "linear", random_state = 0)


# In[34]:


# train model
svc.fit(x_train, y_train)


# In[35]:


y_svm_pred = svc.predict(x_test)


# In[36]:


y_svm_pred


# In[37]:


y_test


# In[38]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[39]:


confusion_matrix(y_test, y_svm_pred)


# In[40]:


accuracy_score(y_test, y_svm_pred)


# In[41]:


# printing the accuracy in percentage
svm_accuracy = accuracy_score(y_test, y_svm_pred)*100
print('Accuracy of our model is equal to ' + str(round(svm_accuracy, 2)) + '%')


# In[ ]:




