# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:20:40 2023

@author: kents
"""

import pandas as pd
import math
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error


#%% Import data

df_original = pd.read_csv(r"C:\Users\kents\OneDrive\Desktop\I-66E_to_Glebe.csv", index_col = 0)

df = df_original[['Year_Record', 'Month_Record', 'Day_Record', 'Day_of_Week', 'Hour_Record', 'Hour_Volume']]

# Drop rows where Hour_Volume equals '000-1'
df = df.drop(index = df.query("Hour_Volume == '000-1'").index)
df = df.astype({'Hour_Volume':'int'})

df['DateTime'] = df.apply(lambda x : datetime(x['Year_Record'] + 2000, x['Month_Record'], x['Day_Record'], x['Hour_Record']), axis = 1)
df = df.set_index('DateTime')

#%% Explore data

#%% Create additional features

#%% Preprocessing

training = df.query("Year_Record < 22")
testing = df.query("Year_Record >= 22")

x = training.drop(columns = ['Hour_Volume'])
y = training['Hour_Volume']

x_test = testing.drop(columns = ['Hour_Volume'])
y_test = testing['Hour_Volume']

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.fit_transform(x)

# Model

n_neighbors = list(range(1,30))
leaf_size = list(range(1,50))
p = [1,2]
weights = ['uniform','distance']

hyperparameters = dict(n_neighbors = n_neighbors, 
                       leaf_size = leaf_size, 
                       p = p, 
                       weights = weights)

knn = KNeighborsRegressor()

clf = RandomizedSearchCV(knn, 
                         hyperparameters, 
                         cv=10, 
                         scoring = 'neg_root_mean_squared_error', 
                         verbose=3, 
                         random_state = 5, 
                         n_iter = 100, 
                         n_jobs = -1)

# Fit the model
best_model = clf.fit(x_scaled, y)

#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
print('Best weight:', best_model.best_estimator_.get_params()['weights'])

cvres = clf.cv_results_
print(max(cvres["mean_test_score"]))


#%% Testing
x_test_scaled = scaler.transform(x_test)

y_test_pred = best_model.predict(x_test_scaled)

print(-1 * math.sqrt(mean_squared_error(y_test, y_test_pred)))

#%% Plotting

pts = 768

x_data = testing.tail(pts).index

plt.plot(x_data, y_test[-pts:], label = 'actual')
plt.plot(x_data, y_test_pred[-pts:], label = 'predicted')
plt.xticks(rotation = 45, ha = 'right')

plt.ylabel("Vehicles")
plt.xlim(left = min(x_data), right = max(x_data))
plt.title("KNN Traffic Prediction")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon = False)

filename = r"C:\Users\kents\OneDrive\Desktop\knn_2022_dec_1.png"

plt.savefig(filename, bbox_inches='tight', dpi=600)