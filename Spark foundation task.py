#!/usr/bin/env python
# coding: utf-8

# ## Name - Dhiraj Chaubey 
# 
# ## The Spark Foundation
# 
# ## Data Science and Business Analytics Intern

# ## Task 1: Prediction using Supervised ML

# In[1]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import metrics


# In[2]:


# Reading the Data 
data = pd.read_csv('http://bit.ly/w-data')
data.head()


# In[3]:


#describing entire data
data.describe(include='all')


# In[4]:


# Check if there any null value in the Dataset
data.isnull().sum()


# ##### There are no null values and hence data cleaning is not required

# ### Data visualization

# In[5]:


##ploting Scatter plot
plt.xlabel('Hours',fontsize=15, fontweight = 'bold')
plt.ylabel('Scores',fontsize=15, fontweight = 'bold')
plt.title('Hours studied vs Score', fontsize=15, fontweight ='bold')
plt.scatter(data.Hours,data.Scores,color='blue')
plt.show()


# #### The above scatter plot indicates positive relationship between Scores and hours studied

# In[6]:


X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values
print(X)
print(Y)


# ## Preparing Data and splitting into train and test sets

# In[7]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size=0.2)


# In[8]:


# We have Splitted Our Data Using 80:20 RULe(PARETO)
print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape  =", X_test.shape)
print("Y test.shape  =", Y_test.shape)


# ## Training the model

# In[9]:


linreg=LinearRegression()


# In[10]:


#Fitting Training Data
linreg.fit(X_train,Y_train)


# In[11]:


print("B0 =",linreg.intercept_,"\nB1 =",linreg.coef_)


# In[12]:


#Plotting on train data
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='red')
line = linreg.coef_*X+linreg.intercept_
plt.plot(X,line,color='black')
plt.xlabel("Hours",fontsize=15, fontweight = 'bold')
plt.ylabel("Scores",fontsize=15, fontweight = 'bold')
plt.title("Regression Line(Train set)",fontsize=15, fontweight = 'bold')
plt.show()


# ## Test data

# In[13]:


#Predicting the Scores for test data
Y_pred=linreg.predict(X_test)
print(Y_pred)


# In[14]:


#Plotting line on test data
plt.plot(X_test,Y_pred,color='black')
plt.scatter(X_test,Y_test,color='red')
plt.xlabel("Hours",fontsize=15, fontweight = 'bold')
plt.ylabel("Scores",fontsize=15, fontweight = 'bold')
plt.title("Regression Line(Test set)",fontsize=15, fontweight = 'bold')
plt.show()


# ## Comparing actual and predicted scores

# In[15]:


Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare


# ## Accuracy of Model

# In[16]:


metrics.r2_score(Y_test,Y_pred)


# #### 94% indicates the above fitted model is Good model

# ## Predicting error

# In[17]:


Mean_Absolute_Error = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Absolute Error = ",Mean_Absolute_Error)


# ## Predicting the score

# In[18]:


Score_prediction = linreg.predict([[9.25]])
print("Predicted score for a student studying 9.25 hours :",Score_prediction)


# ## Conclusion

# ## From the above results we can say that if students studied for 9.25 hours then they may get 93.69 percentage

# In[ ]:




