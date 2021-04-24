#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import streamlit as st 

st.title('Model Deployment: Medical Insurance')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    data = {'CLMSEX':CLMSEX,
            'CLMAGE':CLMAGE,
            }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
# In[2]:


medfraud = pd.read_csv("C:/Users/H P/Desktop/EXCELR PROJECT/Insurance Dataset.csv")
medfraud 


# In[3]:


medfraud.dtypes


# In[4]:


from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


# In[5]:


label_encoder = preprocessing.LabelEncoder()
medfraud['Gender'] = label_encoder.fit_transform(medfraud['Gender'])
medfraud['Cultural_group'] = label_encoder.fit_transform(medfraud['Cultural_group'])
medfraud['ethnicity'] = label_encoder.fit_transform(medfraud['ethnicity'])
medfraud['Admission_type'] = label_encoder.fit_transform(medfraud['Admission_type'])
medfraud['Home or self care,'] = label_encoder.fit_transform(medfraud['Home or self care,'])
medfraud['Surg_Description'] = label_encoder.fit_transform(medfraud['Surg_Description'])
medfraud['Emergency dept_yes/No'] = label_encoder.fit_transform(medfraud['Emergency dept_yes/No'])
medfraud['Abortion'] = label_encoder.fit_transform(medfraud['Abortion'])
medfraud['apr_drg_description'] = label_encoder.fit_transform(medfraud['apr_drg_description'])
medfraud['Age'] = label_encoder.fit_transform(medfraud['Age'])
medfraud['Days_spend_hsptl'] = label_encoder.fit_transform(medfraud['Days_spend_hsptl'])
medfraud


# In[6]:


Hospital = medfraud['Hospital Id']
Hospital.isnull().sum() 


# In[7]:


import statistics
from statistics import mode 


# In[8]:


statistics.mode(medfraud['Hospital Id'])


# In[9]:


medfraud['Hospital Id'] = medfraud['Hospital Id'].fillna(mode)


# In[10]:


Hospital = medfraud['Hospital Id']
Hospital.isnull().sum() 


# In[11]:


Hospital = medfraud['Mortality risk']
Hospital.isnull().sum() 


# In[12]:


statistics.mode(medfraud['Mortality risk'])


# In[13]:


medfraud['Mortality risk'] = medfraud['Mortality risk'].fillna(mode)


# In[14]:


Hospital = medfraud['Mortality risk']
Hospital.isnull().sum() 


# In[15]:


mode1=medfraud['Hospital County'].mode()
medfraud['Hospital County']=medfraud['Hospital County'].fillna(mode1.iloc[0])
mode2=medfraud['Area_Service'].mode()
medfraud['Area_Service']=medfraud['Area_Service'].fillna(mode2.iloc[0])


# In[16]:


medfraud.isna().sum()


# In[17]:


medfraud['Mortality risk']=pd.to_numeric(medfraud['Mortality risk'],errors='coerce')
medfraud['Tot_charg']=pd.to_numeric(medfraud['Tot_charg'],errors='coerce')
medfraud['Tot_cost']=pd.to_numeric(medfraud['Tot_cost'],errors='coerce')
medfraud['ratio_of_total_costs_to_total_charges']=pd.to_numeric(medfraud['ratio_of_total_costs_to_total_charges'],errors='coerce')


# In[18]:


medfraud['Mortality risk']= label_encoder.fit_transform(medfraud['Mortality risk'])
medfraud['Tot_charg']= label_encoder.fit_transform(medfraud['Tot_charg'])
medfraud['Tot_cost']= label_encoder.fit_transform(medfraud['Tot_cost'])
medfraud['ratio_of_total_costs_to_total_charges']= label_encoder.fit_transform(medfraud['ratio_of_total_costs_to_total_charges'])


# In[19]:


medfraud.dtypes



#***************************** USING UNIVARIATE SELECTION FOR FEATURE SELECTION************************************************


# In[38]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[39]:


medfraud['Area_Service']=pd.to_numeric(medfraud['Area_Service'],errors='coerce')
medfraud['Hospital County']=pd.to_numeric(medfraud['Hospital County'],errors='coerce')
medfraud['Hospital Id']=pd.to_numeric(medfraud['Hospital Id'],errors='coerce')


# In[40]:


medfraud['Area_Service']= label_encoder.fit_transform(medfraud['Area_Service'])
medfraud['Hospital County']= label_encoder.fit_transform(medfraud['Hospital County'])
medfraud['Hospital Id']= label_encoder.fit_transform(medfraud['Hospital Id'])
medfraud 


# In[41]:


X = medfraud.iloc[:,0:22] 
X 


# In[42]:


y = medfraud.iloc[:,22]
y 


# In[43]:


bestfeatures = SelectKBest(score_func=chi2, k=22)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[44]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  
print(featureScores.nlargest(22,'Score'))  


# In[45]:


# *********************************************MODEL BUILDING****************************************************************** 


# In[46]:


# 2. MODEL BUILDING


# In[47]:


medfraud 


# In[48]:


med = medfraud.drop(['apr_drg_description','Surg_Description','Abortion','Mortality risk','ethnicity','Days_spend_hsptl'
                    ,'Admission_type','Emergency dept_yes/No','Code_illness','Cultural_group','Gender','ccs_procedure_code','Home or self care,','Age'],axis=1)
med 


# In[49]:


x = med.iloc[:,0:8]
x 


# In[50]:


y = med.iloc[:,8]
y 


# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


classifier = LogisticRegression()
classifier.fit(x,y)



# In[53]:


classifier.coef_


# In[54]:


classifier.predict_proba (x) 


# In[55]:


y_pred = classifier.predict(x)
med["y_pred"] = y_pred
med  


# In[56]:


y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))
new_df = pd.concat([med,y_prob],axis=1)
new_df 


# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)


# In[58]:


pd.crosstab(y_pred,y) 


# In[59]:


accuracy = sum(y==y_pred)/med.shape[0]
accuracy

prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.subheader('Accuracy')
st.write('Accuracy is')