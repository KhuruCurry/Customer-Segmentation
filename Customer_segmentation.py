# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:12:15 2022

@author: Khuru
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential,Input
from tensorflow.keras.utils import plot_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# =============================================================================
# Cramer’s V is a measure of the strength of association between two nominal variables. 
# It ranges from 0 to 1 where:
# 0 indicates no association between the two variables.
# 1 indicates a strong association between the two variables.
# =============================================================================
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher, 
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) 
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# =============================================================================
# In this section, we define a multilayer Perceptron model for binary classification.
# =============================================================================
def simple_two_layer_model(nb_classes,drop_rate=0.2,num_node_hidden_layer=32):
    # Sequential Approach
    model = Sequential()
    model.add(Input(shape=(nb_features)))
    model.add(Dense(num_node_hidden_layer, activation='linear',name='Hidden_Layer_1'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(Dense(nb_classes,activation='softmax',name='Output_Layer'))
    model.summary()
    return model
    
#%% STATICS
CSV_PATH = os.path.join(os.getcwd(),'Train.csv')
JOB_TYPE_ENCODER_PATH = os.path.join(os.getcwd(),'job_type_encoder.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'marital_encoder.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'education_encoder.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'default_encoder.pkl')
HOUSING_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'housing_loan_encoder.pkl')
PERSONAL_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'personal_loan_encoder.pkl')
COMMUNICATION_TYPE_ENCODER_PATH = os.path.join(os.getcwd(),'communication_type_encoder.pkl')
TERM_DEPOSIT_SUBSCRIBED_ENCODER_PATH = os.path.join(os.getcwd(),'term_deposit_subscribed_encoder.pkl')
MONTH_ENCODER_PATH = os.path.join(os.getcwd(),'month_encoder.pkl')
PREV_CAMPAIGN_OUTCOME_ENCODER_PATH = os.path.join(os.getcwd(),'prev_campaign_outcome_encoder.pkl')


CUSTOMER_OHE_PICKLE_PATH = os.path.join(os.getcwd(),'customer_ohe.pkl')
#XSCALED_PATH = os.path.join(os.getcwd(),'x_scaled.pkl')
CUSTOMER_MODEL_SAVE_PATH = os.path.join(os.getcwd(),'customer_model.h5')

# EDA
# STEP 1) Data Loading
df = pd.read_csv(CSV_PATH)

# Step 2) Data Inspection
df.info() # check for NaNs, check which column is categorical
df.describe().T # percentile, mean, min-max, count

plt.figure(figsize=(10,12))
df.boxplot()
plt.show()

cat_column = ['job_type','marital','education','default','housing_loan',
              'personal_loan','communication_type','term_deposit_subscribed','month'
              ,'prev_campaign_outcome']


cont_column = ['customer_age','balance','day_of_month',
               'last_contact_duration','num_contacts_in_campaign',
               'days_since_prev_campaign_contact','num_contacts_prev_campaign']

# to categorical data
for i in cat_column:
    plt.figure()
    sns.countplot(df[i])
    plt.show()
	

# continous data
for i in cont_column:
    plt.figure()
    sns.distplot(df[i])
    plt.show()


df.groupby(['education','term_deposit_subscribed']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')

#%%
# Step 3) DATA CLEANING
# Drop ID column
# convert categorical columns to integers

df.isna().sum()
df.duplicated().sum() # 0 duplicated data


df = df.drop(labels=['id'],axis=1) 
df.info() # double check if these columns has been removed--> 'id' column ,'days_since_prev_campaign_contact'

cont_column = ['customer_age','balance','day_of_month',
               'last_contact_duration','num_contacts_in_campaign',
               'num_contacts_prev_campaign']

# Data Imputation (Remove NaNs, and unique symbols)
df = df.replace(r'^\s*$',np.nan, regex=True)
df.info()

# fill mode for NaNs value
for i in ['marital','personal_loan']:
    df[i].fillna(df[i].mode()[0], inplace=True) 
#df[['customer_age','balance','last_contact_duration','num_contacts_prev_campaign']] = df[['customer_age','balance','last_contact_duration','num_contacts_prev_campaign']].fillna(df[['customer_age','balance','last_contact_duration','num_contacts_prev_campaign']].median())

# fill median for NaNs value
for i in ['customer_age','balance','last_contact_duration','num_contacts_prev_campaign',
          'num_contacts_in_campaign']:
    df[[i]] = df[[i]].fillna(df[[i]].median())

df.info()


#%% Way to convert categorical into integers
paths = [JOB_TYPE_ENCODER_PATH, MARITAL_ENCODER_PATH ,EDUCATION_ENCODER_PATH,
         DEFAULT_ENCODER_PATH , HOUSING_LOAN_ENCODER_PATH, 
         PERSONAL_LOAN_ENCODER_PATH, COMMUNICATION_TYPE_ENCODER_PATH, 
         TERM_DEPOSIT_SUBSCRIBED_ENCODER_PATH, MONTH_ENCODER_PATH,
         PREV_CAMPAIGN_OUTCOME_ENCODER_PATH]

le = LabelEncoder()

for index,i in enumerate(cat_column):
    temp = df[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df[i] = pd.to_numeric(temp,errors='coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)

# 4) Feature Selection

for i in cat_column:
    print(i)
    confussion_mat = pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))

for i in cont_column:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed']))

# Step 5) Data pre-processing
X = df.loc[:,['term_deposit_subscribed','customer_age','balance','day_of_month',
              'last_contact_duration','num_contacts_in_campaign',
              'num_contacts_prev_campaign']]
y = df['term_deposit_subscribed']

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y, axis=-1))

# Need to save ohe model

with open(CUSTOMER_OHE_PICKLE_PATH,'wb') as file:
    pickle.dump(ohe,file)


#%% MODEL DEVELOPMENT

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=123)

nb_features = np.shape(X)[1:]
nb_classes = len(np.unique(y_train))

model = simple_two_layer_model(nb_classes)
 
model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics='acc')
                               
# callbacks

early_stopping_callback = EarlyStopping(monitor='loss',patience=3)


hist = model.fit(X_train,y_train,
                 batch_size=64,
                 validation_data=(X_test,y_test),
                 epochs=30,callbacks=[early_stopping_callback])

#%%Model Evaluation

hist.history.keys()

#To show plots that indicates training loss and validation loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training loss','validation loss'])
plt.show()

#To show plots that indicates training accuracy and validation accuracy
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training accuracy','validation accuracy'])
plt.show()

# from the graph, it shows that the model experiences underfitting
# so more layers/nodes have to be added

y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test),axis=1)

cr = classification_report(y_true,y_pred)
cm = confusion_matrix(y_true, y_pred)
print(cr)
print(cm)

labels = ['0','1']
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
                 
# MODEL SAVING
model.save(CUSTOMER_MODEL_SAVE_PATH)

#%%

plot_model(model,show_shapes=True, show_layer_names=((True)))

