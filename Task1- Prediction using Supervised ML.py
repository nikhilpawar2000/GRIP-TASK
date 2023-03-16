#!/usr/bin/env python
# coding: utf-8

# # Author: PAWAR NIKHIL

# # GRIP@ The Spark Foundation: Data Science & Business Analytics Internship.
# 

# # Task#1- Prediction using Supervised ML
# 
# 

# In[1]:


# Importing all required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 


# In[3]:


# Reading data 
data = pd.read_csv("C:\\Users\\nikhi\\Downloads\\student_scores - student_scores.csv")
print("Data imported successfully")
data.head(15)


# In[4]:


# Checking dimensions and information of the data
print(data)


print("dimensions:")
print(data.shape)


print("Information:")
data.info()


# In[5]:


# Statistical Summary of the data
print("Statistical Summary")
data.describe()


# In[6]:


# Checking missing values
data.isnull().values.any()


# In[7]:


# Scatterplot
data.plot(kind="scatter",x="Hours",y="Scores",color="b")
plt.show()


# In[8]:


# Check Correlation between Hours and scores
data.corr(method="pearson")


# # Linear Regression

# In[9]:


#Splitting data in x and y
x=data.iloc[:, :-1].values
y=data.iloc[:, 1].values


# In[10]:


#The next step is to split this data into training and testing sets.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0) 


# In[11]:


#Train the model
from sklearn.linear_model import LinearRegression  
lr=LinearRegression()  
lr.fit(X_train, y_train) 


# In[12]:


#Plotting the Regression Line
line=lr.coef_*x+lr.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[13]:


# Test the model and check Accuracy
y_pred=lr.predict(X_test)
df=pd.DataFrame({"Actual":y_test,"Predict":y_pred})
print(df)
print("Accuracy:",lr.score(x,y)*100)


# In[14]:


h=9
b=lr.predict([[h]])
print(f"If a student studied for {h} hours/day will score{b}% in exam.")


# # Model Evaluation

# In[15]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("mean squared error:",mean_squared_error(y_test,y_pred))
print("mean absolute error:",mean_absolute_error(y_test,y_pred))
print("R2 score:",r2_score(y_test,y_pred))


# # Thank you
