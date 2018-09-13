import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
# handling null values for temperature columns
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from skopt.space import Real,Integer,Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.datasets import load_svmlight_files
from xgboost.sklearn import XGBRegressor
import json
import codecs
from collections import OrderedDict
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import time


space = {
        'n_estimators': hp.uniform('n_estimators', 100, 1000),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 60, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': 4,
        "xg_reg_alpha": hp.loguniform('xg_reg_alpha',1e-9,1.0),
        "xg_reg_lambda":hp.loguniform('xg_reg_lambda',1e-9,1)

    }


report_Data_beer = OrderedDict()

train_Data = pd.read_csv("Beer Train Data Set.csv")
test_Data = pd.read_csv("Beer Test Data Set.csv")

train_Data = shuffle(train_Data, random_state=2)


train_Data.reset_index(drop=True,inplace=True)
test_Data.reset_index(drop=True,inplace=True)


mean_abv = train_Data["ABV"].mean()

print(mean_abv)

train_Data["ABV"].fillna(mean_abv, inplace=True)

test_mean_abv = test_Data["ABV"].mean()

test_Data["ABV"].fillna(test_mean_abv, inplace=True)


# handling ratings column
train_Data["Ratings"] = train_Data['Ratings'].str.replace(',', '')
train_Data["Ratings"] = train_Data['Ratings'].astype(np.float64)

test_Data["Ratings"] = test_Data['Ratings'].str.replace(',', '')
test_Data["Ratings"] = test_Data['Ratings'].astype(np.float64)

#handling cellar temperature
train_Data["Cellar Temperature"].fillna("35-40",inplace=True)

test_Data["Cellar Temperature"].fillna("35-40",inplace=True)

#handling serving
train_Data["Serving Temperature"].fillna("40-45",inplace=True)

test_Data["Serving Temperature"].fillna("40-45",inplace=True)

#doing one hot encoding on categorical features for train data
style_dataframe = pd.get_dummies(train_Data["Style Name"],sparse=True)
print("style_dataframe",style_dataframe)

glass_ware_dataframe = pd.get_dummies(train_Data["Glassware Used"],sparse=True)

print(len(glass_ware_dataframe))

food_pairing_dataframe = pd.get_dummies(train_Data["Food Paring"],sparse=True)

print(len(food_pairing_dataframe))

train_Data["Cellar Temperature"].fillna("35-40",inplace=True)

cellar_temp_dataframe = pd.get_dummies(train_Data["Cellar Temperature"])

print(len(cellar_temp_dataframe))

cellar_temp_dataframe.rename(columns={"35-40": "c_t1", "40-45": "c_t2","45-50":"c_t3"},inplace=True)

serving_temp_dataframe = pd.get_dummies(train_Data["Serving Temperature"])

print(len(serving_temp_dataframe))

# In[50]:

combined_train_data_category = pd.concat([style_dataframe,cellar_temp_dataframe,food_pairing_dataframe],axis=1)
print(len(combined_train_data_category))
print(combined_train_data_category.columns.values)

#doing the same thing for test data

t_style_dataframe = pd.get_dummies(test_Data["Style Name"],sparse=True)

print(len(t_style_dataframe))

t_glass_ware_dataframe = pd.get_dummies(test_Data["Glassware Used"],sparse=True)


print(len(t_glass_ware_dataframe))

t_food_pairing_dataframe = pd.get_dummies(test_Data["Food Paring"],sparse=True)

print(len(t_food_pairing_dataframe))

t_cellar_temp_dataframe = pd.get_dummies(test_Data["Cellar Temperature"])

print(len(t_cellar_temp_dataframe))

t_cellar_temp_dataframe.rename(columns={"35-40": "c_t1", "40-45": "c_t2","45-50":"c_t3"},inplace=True)

t_serving_temp_dataframe = pd.get_dummies(test_Data["Serving Temperature"])

print(len(t_serving_temp_dataframe))

# In[50]:
t_combined_train_data_category = pd.concat([t_style_dataframe,t_cellar_temp_dataframe,t_food_pairing_dataframe],axis=1)
print(len(t_combined_train_data_category))
print(t_combined_train_data_category.columns.values)

combined_train_data = pd.concat([train_Data[["ABV","Ratings","Brewing Company"]],combined_train_data_category],axis=1)

combined_test_data = pd.concat([test_Data[["ABV","Ratings","Brewing Company"]],t_combined_train_data_category],axis=1)


X_train, X_validtn, y_train, y_validtn = train_test_split(combined_train_data, train_Data["Score"], test_size=0.2,
                                                          random_state=1)

trials = Trials()




def train_xg_boost(params):
    xg_model = XGBRegressor(n_estimators=int(params['n_estimators']), learning_rate=params['eta'], n_jobs=-1,
                            max_depth=int(params['max_depth']), gamma=params['gamma'], colsample_bytree=params['colsample_bytree'],
                            min_child_weight=params['min_child_weight'], reg_alpha=params['xg_reg_alpha'], subsample=params['subsample'],
                            reg_lambda=params['xg_reg_lambda']
                            )

    # In[ ]:
    xg_model.fit(X_train.values, y_train.values)
    training_values = xg_model.predict(X_train.values)
    print(training_values)
    training_rmse = math.sqrt(mean_squared_error(y_train, training_values))
    print("training_rmse", training_rmse)
    validation_values = xg_model.predict(X_validtn.values)
    validation_rmse = math.sqrt(mean_squared_error(y_validtn, validation_values))
    print("validation_rmse", validation_rmse)
    """test_submission = pd.DataFrame()
    test_submission["Score"] = xg_model.predict(combined_test_data)
    test_submission.to_excel('submission4.xlsx', index=False)"""

    return {
        'loss': validation_rmse,
        'status': STATUS_OK,
        'eval_time': time.time(),
    }

print(trials.results)


best = fmin(fn=train_xg_boost, space=space, algo=tpe.suggest,max_evals=100,trials=trials)
                # trials=trials,



