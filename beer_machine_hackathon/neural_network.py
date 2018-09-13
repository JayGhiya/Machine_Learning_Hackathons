import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
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
from skopt import gp_minimize
from collections import OrderedDict

no_of_iterations = Integer(name="no_of_iterations",low=200,high=8000)
learning_rate  = Real(name="learning_rate",low=1e-5,high=1,prior="log-uniform")
layer_1_size = Integer(name='layer_1_size',low=5,high=20)
layer_2_size = Integer(name='layer_2_size',low=5,high=20)
m_alpha = Real(name="m_alpha",low=1e-8,high=1,prior="log-uniform")


dimensions = [no_of_iterations,learning_rate,layer_1_size,layer_2_size,m_alpha]


train_Data = pd.read_csv("C:\\Users\\Jay Ghiya\\Documents\Machine_hack\\beer_competition\\Beer Train Data Set.csv")

test_Data = pd.read_csv("C:\\Users\\Jay Ghiya\\Documents\Machine_hack\\beer_competition\\Beer Test Data Set.csv")

train_Data = shuffle(train_Data, random_state=2)
report_Data_beer = OrderedDict()

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

train_Data["Ratings"] = (train_Data["Ratings"] - train_Data["Ratings"].mean()) / train_Data["Ratings"].std()
train_Data["ABV"] = (train_Data["ABV"] - train_Data["ABV"].mean()) / train_Data["ABV"].std()

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

combined_train_data = pd.concat([train_Data[["ABV","Ratings"]],combined_train_data_category],axis=1)

X_train, X_validtn, y_train, y_validtn = train_test_split(combined_train_data, train_Data["Score"], test_size=0.2,
                                                          random_state=1)


@use_named_args(dimensions=dimensions)
def train_xg_boost(no_of_iterations,learning_rate,layer_1_size,layer_2_size,m_alpha):
    ml_model = MLPRegressor(hidden_layer_sizes=(layer_1_size,layer_2_size),max_iter=no_of_iterations,learning_rate_init=learning_rate,alpha=m_alpha,batch_size=800,random_state=42)
    ml_model.fit(X_train, y_train)
    training_values = ml_model.predict(X_train)
    print(training_values)
    training_rmse = math.sqrt(mean_squared_error(y_train, training_values))
    print("training_rmse", training_rmse)
    validation_values = ml_model.predict(X_validtn)
    validation_rmse = math.sqrt(mean_squared_error(y_validtn, validation_values))
    print("validation_rmse", validation_rmse)
    return validation_rmse



result = gp_minimize(func=train_xg_boost, dimensions=dimensions, n_calls=100)

print("best parameters", result.x)
print("function value", result.fun)

report_Data_beer['best_parameters'] = str(result.x)
report_Data_beer['function_value'] = str(result.fun)

with open('neural_BEER_CONFIG.json', 'wb') as f:
    json.dump(list(report_Data_beer.items()), codecs.getwriter('utf-8')(f), ensure_ascii=False)"""


