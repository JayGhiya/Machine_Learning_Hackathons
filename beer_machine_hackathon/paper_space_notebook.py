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

# In[48]:

#setting the space of gp_minimize


xg_no_estimator = Integer(name="xg_no_estimator",low=50,high=1000)
xg_learning_rate = Real(name="xg_learning_rate",low=0.01,high=1,prior="log-uniform")
xg_max_depth = Integer(name="xg_max_depth",low=0,high=80)
xg_min_child_weight = Integer(name="xg_min_child_weight",low=0 ,high=8)
xg_gamma = Real(name="xg_gamma",low=1e-9,high=0.5,prior="log-uniform")
xg_subsample = Real(name="xg_subsample",low=0.01,high=1.0,prior="uniform")
xg_colsample_bytree = Real(name="xg_colsample_bytree",low=0.01,high=1,prior="uniform")
xg_colsample_bylevel = Real(name="xg_colsample_bylevel",low=0.01,high=1,prior="uniform")
xg_reg_alpha = Real(name="xg_reg_alpha",low=1e-9,high=1.0,prior="log-uniform")
xg_reg_lambda = Real(name="xg_reg_lambda",low=1e-9,high=1,prior="log-uniform")


dimensions = [xg_no_estimator,xg_learning_rate,xg_max_depth,xg_min_child_weight,xg_gamma,xg_subsample,xg_colsample_bytree,xg_colsample_bylevel,xg_reg_alpha,xg_reg_lambda]


report_Data_beer = OrderedDict()

train_Data = pd.read_csv("Beer Train Data Set.csv")
test_Data = pd.read_csv("Beer Test Data Set.csv")

train_Data = shuffle(train_Data, random_state=2)


train_Data.reset_index(drop=True,inplace=True)
test_Data.reset_index(drop=True,inplace=True)

#working on ABV column of beer train and test
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
# In[49]:

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


t_combined_train_data_category = pd.concat([cellar_temp_dataframe],axis=1)
print(len(t_combined_train_data_category))
print(t_combined_train_data_category.columns.values)

# In[51]:
combined_train_data = pd.concat([train_Data[["ABV","Ratings","Brewing Company"]],combined_train_data_category],axis=1)

combined_test_data = pd.concat([test_Data[["ABV","Ratings"]],t_combined_train_data_category],axis=1)



X_train, X_validtn, y_train, y_validtn = train_test_split(combined_train_data, train_Data["Score"], test_size=0.2,
                                                          random_state=1)

eval_set = [(X_train, y_train), (X_validtn, y_validtn)]
@use_named_args(dimensions=dimensions)
def train_xg_boost(xg_no_estimator, xg_learning_rate, xg_max_depth, xg_min_child_weight, xg_gamma, xg_subsample,
                   xg_colsample_bytree, xg_colsample_bylevel, xg_reg_alpha, xg_reg_lambda):

    xg_model = XGBRegressor(n_estimators=xg_no_estimator, learning_rate=xg_learning_rate, n_jobs=-1,
                            max_depth=xg_max_depth, gamma=xg_gamma, colsample_bytree=xg_colsample_bytree,
                            min_child_weight=xg_min_child_weight, reg_alpha=xg_reg_alpha, subsample=xg_subsample,
                            colsample_bylevel=xg_colsample_bylevel, reg_lambda=xg_reg_lambda,
                            )
    print("no_estimators", xg_no_estimator)
    print("learning rate", xg_learning_rate)
    print("max_depth", xg_max_depth)
    print("gamma", xg_gamma)
    print("subsample", xg_subsample)
    print("colsample tree", xg_colsample_bytree)
    print("colsample tree", xg_colsample_bytree)
    print("reg alpha", xg_reg_alpha)
    print("child weight", xg_min_child_weight)
    print("alpha",xg_reg_lambda)


    # In[ ]:

    xg_model.fit(X_train.values, y_train.values,eval_set=eval_set,early_stopping_rounds=10)

    # In[68]:

    training_values = xg_model.predict(X_train.values)
    print(training_values)
    training_rmse = math.sqrt(mean_squared_error(y_train, training_values))
    print("training_rmse", training_rmse)
    validation_values = xg_model.predict(X_validtn.values)
    validation_rmse = math.sqrt(mean_squared_error(y_validtn, validation_values))
    print("validation_rmse", validation_rmse)
    return validation_rmse


result = gp_minimize(func=train_xg_boost, dimensions=dimensions, n_calls=100)

print("best parameters", result.x)
print("function value", result.fun)

report_Data_beer['best_parameters'] = str(result.x)
report_Data_beer['function_value'] = str(result.fun)

with open('BEER_CONFIG.json', 'wb') as f:
    json.dump(list(report_Data_beer.items()), codecs.getwriter('utf-8')(f), ensure_ascii=False)

