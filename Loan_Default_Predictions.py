#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, train_test_split, train_test_split
from sklearn.ensemble import RandomForestClassifier



# ## Data preprocessing
# 
# The first objective is to adequeately process the datasets so as to make the later analysis conveninent

# In[2]:
data_train = pd.read_csv("E:\Machine Learning\Loan_Default_Prediction\loan_def_train (1).csv")
data_test = pd.read_csv("E:\Machine Learning\Loan_Default_Prediction\loan_def_test (1).csv")

print(data_train.head())



# In[3]:


data_train.head()

# Based on common understanding, we define three categories comprising each of the feature initially present in the datasets

Recreation= ["take a trip", "vacation", "wedding", "moving"]
Expense = ["medical bills", "educational expenses", "home improvements"]
Loan_Debt = ["debt consolidation", "business loan", "small business", "major purchase"]
Personal = ["buy a car", "buy house", "renewable energy"]
Other = ["other"]


# In[6]:

def label_purpose(row):
    if row["Purpose"] in Recreation:
        return "Recreation"
    if row["Purpose"] in Expense:
        return "Expense"
    if row["Purpose"] in Loan_Debt:
        return "Loan_Debt"
    if row["Purpose"] in Personal:
        return "Personal"
    if row["Purpose"] in Other:
        return "other"
    

data_train["Purpose_label"] = data_train.apply(lambda row: label_purpose(row), axis = 1)


# In[8]:


data_train.drop("Purpose", axis=1, inplace=True)

data_test["Purpose_label"] = data_test.apply(lambda row: label_purpose(row), axis =1)
data_test.drop("Purpose", axis=1, inplace=True)

# The training and test data are modified such that only the relevant information is present

Id = data_test["Id"]
data_test.drop("Id", axis=1, inplace=True)
y_tr = data_train["Credit Default"]
data_train.drop(["Credit Default", "Id"], axis=1, inplace=True)



data_train["Monthly_amount_left"] =  data_train["Annual Income"]/12 - data_train["Monthly Debt"]
data_train["Residual_debt"] =  data_train["Current Loan Amount"] - data_train["Current Credit Balance"]

data_test["Monthly_amount_left"] =  data_train["Annual Income"]/12 - data_train["Monthly Debt"]
data_test["Residual_debt"] =  data_train["Current Loan Amount"] - data_train["Current Credit Balance"]


data_train["Annual Income"].fillna((data_train["Annual Income"].median()), inplace= True)
data_train["Years in current job"].fillna((data_train["Years in current job"].value_counts().index[2]), inplace= True)
data_train["Credit Score"].fillna((data_train["Credit Score"].median()), inplace= True)
data_train["Bankruptcies"].fillna(data_train["Bankruptcies"].mean(), inplace= True)
data_train["Months since last delinquent"].fillna(data_train["Months since last delinquent"].mean(), inplace= True)
data_train["Monthly_amount_left"].fillna(data_train["Monthly_amount_left"].median(), inplace= True)


# In[18]:


data_test["Annual Income"].fillna((data_test["Annual Income"].median()), inplace= True)
data_test["Years in current job"].fillna((data_test["Years in current job"].value_counts().index[2]), inplace= True)
data_test["Credit Score"].fillna((data_test["Credit Score"].median()), inplace= True)
data_test["Bankruptcies"].fillna(data_test["Bankruptcies"].median(), inplace= True)
data_test["Months since last delinquent"].fillna(data_test["Months since last delinquent"].mean(), inplace= True)
data_test["Monthly_amount_left"].fillna(data_train["Monthly_amount_left"].median(), inplace= True)



data_train.drop("Months since last delinquent", axis=1, inplace= True)
data_test.drop("Months since last delinquent", axis=1, inplace= True)



lst_1 = data_train["Years in current job"].value_counts().index.tolist()
lst_2 = [10,3,2,0.5,5, 1,6,7,2,8,9]
data_train["Years in current job"] = data_train["Years in current job"].replace(lst_1, lst_2)


lst_1 = data_test["Years in current job"].value_counts().index.tolist()
lst_2 = [10,2,3,0.5,5, 1,4,7,6,8,9]
data_test["Years in current job"] = data_test["Years in current job"].replace(lst_1, lst_2)



tr_num = data_train.select_dtypes(include = ["int", "float"]).columns.to_list()
tr_cat = data_train.select_dtypes(include = ["object"]).columns.to_list()
test_num = data_test.select_dtypes(include = ["int", "float"]).columns.to_list()
test_cat = data_test.select_dtypes(include = ["object"]).columns.to_list()

data_train_num = data_train[tr_num]
data_train_cat = data_train[tr_cat]
data_test_num = data_test[test_num]
data_test_cat = data_test[test_cat]


# The redundant columns are removed

cols_to_drop = ["Annual Income", "Monthly Debt", "Current Credit Balance", "Current Loan Amount"]

data_train.drop(cols_to_drop, axis=1, inplace=True)
data_test.drop(cols_to_drop, axis=1, inplace=True)


for col in cols_to_drop:
    tr_num.remove(col)
    test_num.remove(col)

# Copy of original training and test dataset

X_tr =  data_train.copy()
X_ts =  data_test.copy()

Q1_ai =  data_train["Residual_debt"].quantile(0.25)
Q3_ai =  data_train["Residual_debt"].quantile(0.75)

iqr_ai = Q3_ai - Q1_ai

up_lim_ai = Q3_ai + 1.5*iqr_ai
low_lim_ai = Q1_ai - 1.5*iqr_ai


Q1_cs =  data_train["Credit Score"].quantile(0.25)
Q3_cs =  data_train["Credit Score"].quantile(0.75)

iqr_cs = Q3_cs - Q1_cs

up_lim_cs = Q3_cs + 1.5*iqr_cs
low_lim_cs = Q1_cs - 1.5*iqr_cs

Q1_md =  data_train["Monthly_amount_left"].quantile(0.25)
Q3_md =  data_train["Monthly_amount_left"].quantile(0.75)

iqr_md = Q3_md - Q1_md

up_lim_md = Q3_md + 1.5*iqr_md
low_lim_md = Q1_md - 1.5*iqr_md

Q1_cp =  data_train["Number of Credit Problems"].quantile(0.25)
Q3_cp =  data_train["Number of Credit Problems"].quantile(0.75)

iqr_cp = Q3_cp - Q1_cp

up_lim_cp = Q3_cp + 1.5*iqr_cp
low_lim_cp = Q1_cp - 1.5*iqr_cp

Q1_tl =  data_train["Tax Liens"].quantile(0.25)
Q3_tl =  data_train["Tax Liens"].quantile(0.75)

iqr_tl = Q3_tl - Q1_tl

up_lim_tl = Q3_tl + 1.5*iqr_tl
low_lim_tl = Q1_tl - 1.5*iqr_tl

Q1_mc =  data_train["Maximum Open Credit"].quantile(0.25)
Q3_mc =  data_train["Maximum Open Credit"].quantile(0.75)

iqr_mc = Q3_mc - Q1_mc

up_lim_mc = Q3_mc + 1.5*iqr_mc
low_lim_mc = Q1_mc - 1.5*iqr_mc

Q1_oa =  data_train["Number of Open Accounts"].quantile(0.25)
Q3_oa =  data_train["Number of Open Accounts"].quantile(0.75)

iqr_oa = Q3_oa - Q1_oa

up_lim_oa = Q3_oa + 1.5*iqr_oa
low_lim_oa = Q1_oa - 1.5*iqr_oa

Q1_ch =  data_train["Years of Credit History"].quantile(0.25)
Q3_ch =  data_train["Years of Credit History"].quantile(0.75)

iqr_ch = Q3_ch - Q1_ch

up_lim_ch = Q3_ch + 1.5*iqr_ch
low_lim_ch = Q1_ch - 1.5*iqr_ch


c1 = data_train["Residual_debt"] > up_lim_ai
c10 = data_train["Residual_debt"] < low_lim_ai
c2 = data_train["Credit Score"] < low_lim_cs
c3 = data_train["Credit Score"] > up_lim_cs
c4 = data_train["Monthly_amount_left"] > up_lim_md
c5 = data_train["Number of Credit Problems"] > 4
c6 = data_train["Tax Liens"] > 5
c7 = data_train["Maximum Open Credit"] > up_lim_mc
c8 = data_train["Number of Open Accounts"] > up_lim_oa
c9 = data_train["Years of Credit History"] > up_lim_ch


# In[36]:


Q1_ai_t =  data_test["Residual_debt"].quantile(0.25)
Q3_ai_t =  data_test["Residual_debt"].quantile(0.75)

iqr_ai_t = Q3_ai_t - Q1_ai_t

up_lim_ai_t = Q3_ai_t + 1.5*iqr_ai_t
low_lim_ai_t = Q1_ai_t - 1.5*iqr_ai_t

Q1_cs_t =  data_test["Credit Score"].quantile(0.25)
Q3_cs_t =  data_test["Credit Score"].quantile(0.75)

iqr_cs_t = Q3_cs_t - Q1_cs_t

up_lim_cs_t = Q3_cs_t + 1.5*iqr_cs_t
low_lim_cs_t = Q1_cs_t - 1.5*iqr_cs_t

Q1_md_t =  data_test["Monthly_amount_left"].quantile(0.25)
Q3_md_t =  data_test["Monthly_amount_left"].quantile(0.75)

iqr_md_t = Q3_md_t - Q1_md_t

up_lim_md_t = Q3_md_t + 1.5*iqr_md_t
low_lim_md_t = Q1_md_t - 1.5*iqr_md_t

Q1_cp_t =  data_test["Number of Credit Problems"].quantile(0.25)
Q3_cp_t =  data_train["Number of Credit Problems"].quantile(0.75)

iqr_cp_t = Q3_cp_t - Q1_cp_t

up_lim_cp_t = Q3_cp_t + 1.5*iqr_cp_t
low_lim_cp_t = Q1_cp_t - 1.5*iqr_cp_t

Q1_tl_t =  data_test["Tax Liens"].quantile(0.25)
Q3_tl_t =  data_test["Tax Liens"].quantile(0.75)

iqr_tl_t = Q3_tl_t - Q1_tl_t

up_lim_tl_t = Q3_tl_t + 1.5*iqr_tl_t
low_lim_tl_t = Q1_tl - 1.5*iqr_tl_t

Q1_mc_t =  data_test["Maximum Open Credit"].quantile(0.25)
Q3_mc_t =  data_test["Maximum Open Credit"].quantile(0.75)

iqr_mc_t = Q3_mc_t

up_lim_mc_t = Q3_mc_t + 1.5*iqr_mc_t
low_lim_mc_t = Q1_mc_t - 1.5*iqr_mc_t

Q1_oa_t =  data_test["Number of Open Accounts"].quantile(0.25)
Q3_oa_t = data_test["Number of Open Accounts"].quantile(0.75)

iqr_oa_t = Q3_oa_t - Q1_oa_t

up_lim_oa_t = Q3_oa_t + 1.5*iqr_oa_t
low_lim_oa_t = Q1_oa_t - 1.5*iqr_oa_t

Q1_ch_t =  data_test["Years of Credit History"].quantile(0.25)
Q3_ch_t =  data_test["Years of Credit History"].quantile(0.75)

iqr_ch_t = Q3_ch_t - Q1_ch_t

up_lim_ch_t = Q3_ch_t + 1.5*iqr_ch_t
low_lim_ch_t = Q1_ch_t - 1.5*iqr_ch_t

c1_t= data_test["Residual_debt"] > up_lim_ai_t
c10_t= data_test["Residual_debt"] < low_lim_ai_t
c2_t = data_test["Credit Score"] < low_lim_cs_t
c3_t = data_test["Credit Score"] > up_lim_cs_t
c4_t= data_test["Monthly_amount_left"] > up_lim_md_t
c5_t = data_test["Number of Credit Problems"] > 4
c6_t = data_test["Tax Liens"] > 5
c7_t = data_test["Maximum Open Credit"] > up_lim_mc_t
c8_t = data_test["Number of Open Accounts"] > up_lim_oa_t
c9_t = data_test["Years of Credit History"] > up_lim_ch_t


# In[37]:


X_tr.loc[c1, "Residual_debt"] = up_lim_ai
X_tr.loc[c10, "Residual_debt"] = low_lim_ai
X_tr.loc[c2, "Credit Score"]  = low_lim_cs
X_tr.loc[c3, "Credit Score"] =  up_lim_cs 
X_tr.loc[c4, "Monthly_amount_left"] =  up_lim_md
X_tr.loc[c7, "Maximum Open Credit"] = up_lim_mc
X_tr.loc[c8, "Number of Open Accounts"] = up_lim_oa
X_tr.loc[c9, "Years of Credit History"] = up_lim_ch


# In[38]:


X_ts.loc[c1_t, "Residual_debt"] = up_lim_ai_t
X_ts.loc[c10_t, "Residual_debt"] = low_lim_ai_t
X_ts.loc[c2_t, "Credit Score"]  = low_lim_cs_t
X_ts.loc[c3_t, "Credit Score"] =   up_lim_cs_t 
X_ts.loc[c4_t, "Monthly_amount_left"] =  up_lim_md_t
X_ts.loc[c7_t, "Maximum Open Credit"] = up_lim_mc_t
X_ts.loc[c8_t, "Number of Open Accounts"] = up_lim_oa_t
X_ts.loc[c9_t, "Years of Credit History"] = up_lim_ch_t


df_train = pd.get_dummies(X_tr, columns = tr_cat, drop_first= True)
df_test = pd.get_dummies(X_ts, columns = tr_cat, drop_first= True)


ct_sc = ColumnTransformer([('scaler', StandardScaler(), tr_num)], remainder='passthrough')


from collections import Counter
from imblearn.over_sampling import SMOTE

# In[50]:


oversample = SMOTE()


# In[51]

X_train, X_test, y_train, y_test = train_test_split(df_train, y_tr, test_size = 0.2, random_state=0)


X_train_sc = ct_sc.fit_transform(X_train)
X_test_sc = ct_sc.transform(X_test)

X_sm, y_sm = oversample.fit_resample(X_train_sc, y_train)

# ## Model evalution with confusion matrix

# In[74]:

cls_rf  = RandomForestClassifier()

model_RF = cls_rf.fit(X_sm, y_sm)



y_pred_RF = model_RF.predict(X_test_sc)

import pickle
pickle.dump(model_RF, open("model.pkl", "wb"))
