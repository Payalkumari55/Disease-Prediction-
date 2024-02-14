#!/usr/bin/env python
# coding: utf-8

# In[49]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


# # Training Dataset

# In[2]:


#training dataset
df = pd.read_csv('C:/Users/ranji/Desktop/python/project/Code Alpha project/Disease prediction/dataset/Training.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df.columns[df.isnull().any()]


# In[9]:


df = df.drop('Unnamed: 133',axis=1)


# In[10]:


df.shape


# In[11]:


#countplot
plt.figure(figsize=(10,4))
sns.countplot(data=df,x='prognosis')
plt.xticks(rotation=90)
plt.xlabel('Diseases');


# In[12]:


num = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 
       'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
       'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 
       'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 
       'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 
       'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
       'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
       'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
       'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
       'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 
       'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
       'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 
       'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 
       'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
       'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
       'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 
       'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 
       'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 
       'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
       'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 
       'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 
       'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
       'blister', 'red_sore_around_nose', 'yellow_crust_ooze']


# In[13]:


#correlation
df[num].corr()


# In[14]:


#heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data=df[num].corr(),cmap='coolwarm')


# In[15]:


X = df.drop('prognosis',axis=1)
y = df['prognosis']


# # Model Training

# In[16]:


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Random Forest Classifier

# In[17]:


rf_model = RandomForestClassifier(n_estimators=100,max_features=85,random_state=42)


# In[18]:


def model_train_test(model,X_train,y_train,X_test,y_test):
    
    #model training
    model.fit(X_train,y_train)
    
    #predict
    pred = model.predict(X_test)
    
    #accuracy score
    print("accuracy score = ",accuracy_score(y_test,pred))
    
    #classification report
    print("\n Classification report")
    print(classification_report(y_test,pred))


# In[19]:


model_train_test(rf_model,X_train, y_train, X_test, y_test)


# # Gradient Boosting Algorithm

# In[20]:


Boosting_model = GradientBoostingClassifier(n_estimators=250)


# In[ ]:


model_train_test(Boosting_model,X_train, y_train, X_test, y_test)


# # Decision Tree Classifier

# In[22]:


dt_model = DecisionTreeClassifier()


# In[23]:


model_train_test(dt_model,X_train, y_train, X_test, y_test)


# # Testing Dataset

# In[24]:


#testing dataset
df_test = pd.read_csv('C:/Users/ranji/Desktop/python/project/Code Alpha project/Disease prediction/dataset/Testing.csv')


# In[25]:


df_test.head()


# In[26]:


df_test.shape


# In[27]:


#testing dataset percentage 
print("Testing Dataset percentage = ",100 * len(df_test)/(len(df)+len(df_test)))


# In[28]:


df_test.describe()


# In[29]:


df_test.columns[df.isnull().any()]


# In[30]:


df_test.columns


# In[31]:


X = df_test.drop('prognosis',axis=1)
y = df_test['prognosis']


# # Model Testing

# In[32]:


def test_accuracy(model,X):
    
    #predict
    pred = model.predict(X)
    
    #accuracy score
    print("accuracy score = ",accuracy_score(y,pred))
    
    #classification report
    print("\n",classification_report(y,pred))


# In[33]:


#Random Forest Classifier
test_accuracy(rf_model,X)


# In[34]:


#Gradient Boosting Algorithm
test_accuracy(Boosting_model,X)


# In[35]:


#Decision Tree Classifier
test_accuracy(dt_model,X)


# # Exploratory Data Analysis (EDA)

# In[38]:


# Visualize the distribution of symptoms
sns.countplot(data=df, x='prognosis')
plt.xticks(rotation=90)
plt.title('Distribution of Prognosis')
plt.show()


# In[66]:


#checking for missing values
sns.heatmap(df.isnull())


# In[68]:


#correlation matrix
correlation = df.corr()
print(correlation)


# In[ ]:


#plotting pair plots for the data
sns.pairplot(df, hue="yellow_crust_ooze")
plt.show()


# In[ ]:


#plotting the data distribution plots
df.hist(figsize=(17,14))
plt.show


# In[42]:


# Assuming 'X' contains the symptoms and 'y' contains the prognosis
X = df.drop(columns=['prognosis'])
y = df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[51]:


# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # Confusion Matrix

# In[52]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[56]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Oranges_r', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[57]:


# Example: Visualize feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Important Features')
plt.show()


# # Making Prediction

# In[ ]:


prediction = model.predict(x_test)
print(prediction)


# In[ ]:


accuracy = accuracy_score(prediction, y_test)
print(accuracy)


# In[ ]:




