#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:32:13 2017

@author: Vinay
"""

import os
os.chdir('/Users/Vinay/Projects/Kaggle titanic')
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None  # default='warn'



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('gender_submission.csv')
y = train['Survived']
del train['Survived']
total_data = train.append(test)


#Categorical variables == pclass, sex, embarked
#One hot encode these guys

cat_columns = ['Pclass','Sex','Embarked']
for col in cat_columns:
    data1 = total_data[[col]]
    np_indices = np.argwhere((np.array(list(data1[col]))) == 'nan')
    np_indices = [x[0] for x in np_indices]
    data1.iloc[np_indices,0] = 'nan'
    data = list(data1[col])
    le = LabelEncoder()
    le.fit(data)
    temp = le.transform(data)
    temp = temp.reshape((len(temp),1))    
    enc = OneHotEncoder(sparse=False)
    enc.fit(temp)
    temp = enc.transform(temp)    
    temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data1[col]
            .value_counts().index])
    temp=temp.set_index(total_data.index.values)
    total_data = pd.concat([total_data,temp],axis = 1)
    del enc
    del le

#Removing nans from numeric columns age, sibsp, parch, fare
num_columns = ['Age','SibSp','Parch','Fare']
for col in num_columns:
    data = list(total_data[col])
    data_str = [str(e) for e in data]
    mean_value = np.nanmean(data)
    new_data = [str(mean_value) if v == 'nan' else v for v in data_str]    
    data2 = [float(e) for e in new_data]
    total_data[col+'_modified'] = data2

#dummy varaible (1/0). 1 if there is a cabin allocated, 0 if not              
total_data['cabin_yes'] = [1 if type(v) == str else 0 for v in list(total_data['Cabin'])]

#Subsetting relevant data
df2 = total_data.iloc[:,list(range(11,25))]
# Check df2.columns.values

train_new = df2.iloc[list(range(0,891)),:]
test_new = df2.iloc[list(range(891,1309)),:]

rf = RandomForestClassifier(n_estimators = 500, oob_score=True)
rf.fit(train_new,y)
#rf.oob_score_ == 0.81369
prediction = rf.predict(test_new)
sample['Survived'] = prediction
sample.to_csv('sub_vanilla_rf_1.csv',index=False)

# This submission scored 0.75 on private leaderboard with oob score as 0.81 
# meaning the distribution of dependent is different in both datasetss



