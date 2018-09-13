# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 19:51:55 2018
@author: Jay Ghiya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# handling null values for temperature columns
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from skopt.space import Real,Integer,Categorical
from sklearn.model_selection import train_test_split
from  sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from skopt.utils import use_named_args
from skopt import forest_minimize


train_Data = pd.read_csv("C:\\Users\\Jay Ghiya\\Documents\Machine_hack\\beer_competition\\Beer Train Data Set.csv")
test_Data = pd.read_csv("C:\\Users\\Jay Ghiya\\Documents\Machine_hack\\beer_competition\\Beer Test Data Set.csv")

train_Data = shuffle(train_Data, random_state=2)

#declaring spaces
rf_no_estimator = Integer(name="rf_no_estimator",low=70,high=500)
rf_max_features = Categorical(name="rf_max_features",categories=["auto","sqrt","log2"])
rf_max_depth = Integer(name="rf_max_depth",low=10,high=80)
rf_min_samples_split = Integer(name="rf_min_samples_split",low=3 ,high=10)
rf_min_samples_leaf = Integer(name="rf_min_samples_leaf",low=3 ,high=10)
rf_bootstrap = Categorical(name="rf_bootstrap",categories=[True, False])
rf_min_impurity_decrease = Real(name="rf_min_impurity_decrease",low=1e-9,high=0.5,prior="log-uniform")


dimensions = [rf_no_estimator,rf_max_features,rf_max_depth,rf_min_samples_split,rf_min_samples_leaf,rf_bootstrap,rf_min_impurity_decrease]

print("null values in test data", test_Data.isnull().sum())
print("length")

# printing column names
print(train_Data.columns.values)
print(len(train_Data))
print("score_null", train_Data["Score"].isnull().sum())

# deciding features for baseline model\
print("food", len(train_Data["Food Paring"].value_counts()))

print("Score", len(train_Data["Score"].value_counts()))
print("Score", train_Data["Score"].value_counts())

print("Glassware Used", len(train_Data["Glassware Used"].value_counts()))
print("data type of Glassware Used:", train_Data["Glassware Used"].dtype)

print("Total ABV:", len(train_Data["ABV"].value_counts()))
print("data type of ABV:", train_Data["ABV"].dtype)

print("total brewing companies", len(train_Data["Brewing Company"].value_counts()))
print("data type of brewing companies", train_Data["Brewing Company"].dtype)

print("total beer names", len(train_Data["Beer Name"].value_counts()))
print("data type of beer names", train_Data["Beer Name"].dtype)

print("total ratings", len(train_Data["Ratings"].value_counts()))
print("data type of total ratings", train_Data["Ratings"].dtype)

print("style name", len(train_Data["Style Name"].value_counts()))
print("data type of style name", train_Data["Style Name"].dtype)

print("Cellar temperature", len(train_Data["Cellar Temperature"].value_counts()))
print("data type of cellar temperature", train_Data["Cellar Temperature"].dtype)

print("Serving temperature", len(train_Data["Serving Temperature"].value_counts()))
print("data type of Serving temperature", train_Data["Serving Temperature"].dtype)

print("null values for selected features:", train_Data[
    ["ABV", "Brewing Company", "Beer Name", "Ratings", "Cellar Temperature", "Serving Temperature",
     "Glassware Used"]].isnull().sum())

total_len = len(train_Data)
abv_null_val = train_Data["ABV"].isnull().sum()


def percent(num1, num2):
    num1 = float(num1)
    num2 = float(num2)
    percentage = '{0:.2f}'.format((num1 / num2 * 100))
    return percentage


print(percent(abv_null_val, total_len))

des_abv = train_Data["ABV"].describe()

print(des_abv)

mean_abv = train_Data["ABV"].mean()

print(mean_abv)

train_Data["ABV"].fillna(mean_abv, inplace=True)

test_mean_abv = test_Data["ABV"].mean()

test_Data["ABV"].fillna(test_mean_abv, inplace=True)

# let us check how are the values of serving and cellar temperature
print(train_Data["Cellar Temperature"].head(n=10))
print(train_Data["Serving Temperature"].head(n=10))
print(train_Data["Ratings"].head(n=10))

# handling ratings column
train_Data["Ratings"] = train_Data['Ratings'].str.replace(',', '')
train_Data["Ratings"] = train_Data['Ratings'].astype(np.float64)

test_Data["Ratings"] = test_Data['Ratings'].str.replace(',', '')
test_Data["Ratings"] = test_Data['Ratings'].astype(np.float64)

# handling style column
train_Data["Style Name"] = train_Data["Style Name"].astype('category')
train_Data["Style_Name_Modified"] = train_Data["Style Name"].cat.codes

train_Data["Beer Name"] = train_Data["Beer Name"].astype('category')
train_Data["Beer_Name_Modified"] = train_Data["Beer Name"].cat.codes

test_Data["Style Name"] = test_Data["Style Name"].astype('category')
test_Data["Style_Name_Modified"] = test_Data["Style Name"].cat.codes

train_Data["Brewing Company"] = train_Data["Brewing Company"].astype('category')
train_Data["Brewing_Company_Modified"] = train_Data["Brewing Company"].cat.codes


print(train_Data["Style_Name_Modified"].head(n=10))

train_Data["Glassware Used"] = train_Data["Glassware Used"].astype('category')
train_Data["Glassware_Used_Modified"] = train_Data["Glassware Used"].cat.codes

train_Data["Food Paring"] = train_Data["Food Paring"].astype('category')
train_Data["Food_Paring_Modified"] = train_Data["Food Paring"].cat.codes

test_Data["Glassware Used"] = test_Data["Glassware Used"].astype('category')
test_Data["Glassware_Used_Modified"] = test_Data["Glassware Used"].cat.codes


train_Data["Cellar Temperature"].fillna("35-40",inplace=True)
train_Data["Cellar Temperature"] = train_Data["Cellar Temperature"].astype('category')
train_Data["Cellar_Temperature_modified"] = train_Data["Cellar Temperature"].cat.codes

train_Data["Serving Temperature"].fillna("40-45",inplace=True)
train_Data["Serving Temperature"] = train_Data["Serving Temperature"].astype('category')
train_Data["Serving_Temperature_modified"] = train_Data["Serving Temperature"].cat.codes



def post_process_temp_columns(temp_range):
    if ((temp_range is not None) and (temp_range is not math.nan)):
        temp_range = str(temp_range)
        list_temp = temp_range.split("-")
        var1 = 0
        var2 = 0
        if (len(list_temp) == 2):
            # converting into floats
            var1 = float(list_temp[0])
            var2 = float(list_temp[1])

        return ((var1 + var2) / 2)


"""
train_Data["Cellar_Temperature_modified"] = train_Data["Cellar Temperature"].apply(post_process_temp_columns,
mean_cellar_Temp_modified = train_Data["Cellar_Temperature_modified"].mean()
train_Data["Cellar_Temperature_modified"].replace(to_replace=0, value=mean_cellar_Temp_modified, inplace=True)

test_Data["Cellar_Temperature_modified"] = test_Data["Cellar Temperature"].apply(post_process_temp_columns,
                                                                                 convert_dtype=False)
test_mean_cellar_Temp_modified = test_Data["Cellar_Temperature_modified"].mean()

test_Data["Cellar_Temperature_modified"].replace(to_replace=0, value=mean_cellar_Temp_modified, inplace=True)

test_Data["Serving_Temperature_modified"] = test_Data["Serving Temperature"].apply(post_process_temp_columns,
                                                                                   convert_dtype=False)
test_mean_serving_Temp_modified = test_Data["Serving_Temperature_modified"].mean()
test_Data["Serving_Temperature_modified"].replace(to_replace=0, value=mean_cellar_Temp_modified, inplace=True)



train_Data["Serving_Temperature_modified"] = train_Data["Serving Temperature"].apply(post_process_temp_columns,
                                                                                     convert_dtype=False)

train_Data = train_Data[train_Data["Serving_Temperature_modified"] != 0]
"""

features = ["ABV","Ratings", "Style_Name_Modified","Food_Paring_Modified","Cellar_Temperature_modified","Brewing Company"]

X_train, X_validtn, y_train, y_validtn = train_test_split(train_Data[features], train_Data["Score"], test_size=0.2,
                                                          random_state=1)

@use_named_args(dimensions=dimensions)
def train_random_forest(rf_no_estimator,rf_max_features,rf_max_depth,rf_min_samples_split,rf_min_samples_leaf,rf_bootstrap,rf_min_impurity_decrease):
    print("rf_no_estimator", rf_no_estimator)

    print("rf_max_features", rf_max_features)

    print("rf_max_Depth", rf_max_depth)

    print("rf_min_samples_split", rf_min_samples_split)

    print("rf_min_samples_leaf", rf_min_samples_leaf)

    print("rf_min_sample",rf_min_impurity_decrease)

    print("bootstrap",rf_bootstrap)

    #print("rf_oob_score",rf_oob_score)

    model = RandomForestRegressor(n_estimators=rf_no_estimator, max_depth=rf_max_depth,
                                  min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf,
                                  max_features=rf_max_features,n_jobs=-1,bootstrap=rf_bootstrap,min_impurity_decrease=rf_min_impurity_decrease)

    model.fit(X_train[features], y_train)

    X_train["predicted_score"] = model.predict(X_train[features])

    print("training error", math.sqrt(mean_squared_error(y_train, X_train["predicted_score"])))

    X_validtn["predicted_score"] = model.predict(X_validtn[features])

    val_score = math.sqrt(mean_squared_error(y_validtn, X_validtn["predicted_score"]))

    print("validation  error", val_score)

    return val_score

    # test_submission = pd.DataFrame()

    # test_submission["Score"] = model.predict(test_Data[features])

    # my_submission = pd.DataFrame({'Score': test_Data["Score"]})
    # you could use any filename. We choose submission here
    # test_submission.to_excel('submission4.xlsx', index=False)

    # rmses = cross_val_score(model,train_Data[features],train_Data["Score"],cv=5)

    # print(rmses)

#val_score = train_random_forest(400,"log2",60,8,5,True,0.000010)
#print("validation course",val_score)

result = forest_minimize(func=train_random_forest,dimensions=dimensions,n_calls=100,base_estimator="RF")

print("minimum",result.x)

print("function_value",result.fun)

