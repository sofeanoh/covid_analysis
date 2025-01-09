#%%#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
from time_series_helper import WindowGenerator
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import Sequential, layers, metrics, losses, regularizers
#%%

# Defining constants
DATASETS_PATH = os.path.join(os.getcwd(), os.pardir, 'datasets')
TRAIN_PATH = os.path.join(DATASETS_PATH, 'cases_malaysia_train.csv')
TEST_PATH = os.path.join(DATASETS_PATH, 'cases_malaysia_test.csv')

#%%
#Data Loading
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
 #%%
 
# Data Exploration
print(train_df.head())
print(test_df.head())

# Data Inspection
print(train_df.info())
print(test_df.info())
print("\nmissing values in train: \n", train_df.isna().sum())
print("\nduplicate values in train: ", train_df.duplicated().sum())

# inspect cases_new column
print(train_df['cases_new'].unique())

# found unique value " " and "?" in cases_new column but apparantly it doesnt detect the space as a missing value
weird_cases_new = train_df[train_df['cases_new'] == " "]
weird_cases_new

######### COMMENTS ###########

# We cannot simply remove the weird_cases_new rows because these are time series task.

# Also, notice that all cluster columns are missing 50% of the data. As the starter, i will remove them first.

##############################

#%%
# remove all cluster columns
train_df_copy = train_df.copy()
cluster_columns = ['cluster_import', 'cluster_religious', 'cluster_community', 'cluster_highRisk', 'cluster_education', 'cluster_detentionCentre', 'cluster_workplace']
train_df_copy.drop(columns=cluster_columns, inplace=True)
print(train_df_copy.head())

# reinspect the data after removing the NaN columns
print(train_df_copy.info())

# to investigate the method for imputation of cases_new, i will understand the pattern of the data first

# Data visualisation
date = pd.to_datetime(train_df_copy.pop('date'), dayfirst=True)
print(train_df_copy.head())

print(date)

#%%

# #plot date against cases_new, since case_new has null value and is of type object, so need to plot scatter plot instead
# replace train_df_copy['cases_new'] == " " | train_df_copy['cases_new'] == "?" with np.nan first
# get the index of row where train_df_copy['cases_new'] is NaN
train_df_copy['cases_new'] = train_df_copy['cases_new'].replace(' ', np.nan)
train_df_copy['cases_new'] = train_df_copy['cases_new'].replace('?', np.nan)
train_df_copy['cases_new'] = train_df_copy['cases_new'].astype(float)
nan_cases_new = train_df_copy[train_df_copy['cases_new'].isna()].index
plot_sales = train_df_copy['cases_new']
plot_sales.index = date
_ = plot_sales.plot()
# mark index for nan_cases_new index on plot to see if there is any pattern
for i in nan_cases_new:
    plt.axvline(date[i], color='r', linestyle='--')
plt.show()

# i dont use LOCF or NOCB imputation technique because the data is showing changes in direction, instead i will use spline nterpolation
# impute cases_new with spline interpolation
train_df_copy['cases_new'].interpolate(method='spline', order=3, inplace=True)
print(train_df_copy['cases_new'].isna().sum()) # succesfully remove nan values

# replot the graph
plot_sales = train_df_copy['cases_new']
plot_sales.index = date
_ = plot_sales.plot()
for i in nan_cases_new:
    plt.axvline(date[i], color='r', linestyle='--') # to show the pattern before and after imputation
plt.show()

#%%
# Data Visualisation
plt.figure(figsize=(30, 30))
plot_cols = train_df_copy.columns
plot_features = train_df_copy[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)
plt.show()

train_df_copy.describe().transpose()

######################## COMMENTS ########################

# All of the data has higher std than mean, meaning it there could be outliers, 
# however as this is a time series data where the cases are indipendent from one to other days 
# (a person could has been infected on day 1, but only developing symptoms and visiting the clinic on day 4th of infection.), 
# so we will just ignore it.

##################################################

#%%
#################### Preprocess the test Data #######################

print(test_df.info())

print(test_df['cases_new'].unique())

# i personally think we are interpolating test_df['cases_new'] strictly for ground truth purpose only. as we can actually use our model to predict the nan value on the test_data (but since we dont have
# ground truth data for that NaN value to compare our prediction with, thats why we need to interpolate that specific row of NaN value in the test_df['cases_new'])
test_df_copy = test_df.copy()
test_df_copy['cases_new'] = test_df_copy['cases_new'].interpolate(method='spline', order=3)
print(test_df_copy['cases_new'].isna().sum()) # succesfully remove nan values

# remove the same columns
test_df_copy.drop(columns=cluster_columns, inplace=True)
date_test = pd.to_datetime(test_df_copy.pop('date'), dayfirst=True)

#%%
# I wont split the data into train and val as there are only 680 data points
print(train_df_copy.shape)
print(test_df_copy.shape)

#%%

# data normalisation
scaler = MinMaxScaler() # because we have 'outliers' (but is valid 'outliers') so using standard scaler or manual normalisation will shrink the data
scaled_train = scaler.fit_transform(train_df_copy)
scaled_test = scaler.transform(test_df_copy)

# revert back to dataframe
train_df_copy = pd.DataFrame(scaled_train, columns=train_df_copy.columns)
test_df_copy = pd.DataFrame(scaled_test, columns=test_df_copy.columns)
print(train_df_copy.shape)
print(test_df_copy.shape)
print(train_df_copy.head())

#%%
############# Preparing the Window ############################

#single output single time step -- use 'cases_new' as prediction
single_output_window = WindowGenerator(30, 30, 1, train_df_copy, val_df=None, test_df=test_df_copy, label_columns=['cases_new']) #input_width, label_width, shift (offset), train, val, test, prediction(can be single label or multilabel, but keep in mind it is in list form)
#plot the single_output_window with date as index, and cases_new as y axis
single_output_window.plot(plot_col='cases_new')
plt.xlabel('Day') # because our window size is split based on day here

########################################################

#%%
for inputs, labels in single_output_window.train.take(1):
  inputs_shape = inputs.shape
  labels_shape = labels.shape
  print(f'Inputs shape (batch, time, features): {inputs_shape}')
  print(f'Labels shape (batch, time, features): {labels_shape}')
  
  #%%
  # Model development
# Configuring MLFlow here
# change working directory for mlruns purpose because all of my capstones folder are under capstone folder where the mlruns is in
os.chdir(os.path.join(os.getcwd(), os.pardir))
os.chdir(os.path.join(os.getcwd(), os.pardir))
print(os.getcwd())
# %%
mlflow.set_experiment('covid_forecast')
# %%
with mlflow.start_run():
    # defining the model
    model = Sequential()
    model.add(layers.LSTM(32, return_sequences=True, input_shape=single_output_window.train.element_spec[0].shape[1:]))
    model.add(layers.Dense(1)) #since predicting one output only
    model.summary()
    
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
MAX_EPOCHS = 20
history_single = model.fit(single_output_window.train, validation_data=single_output_window.test, epochs=MAX_EPOCHS, callbacks=[early_stopping]) # no need to split to x, y because our wide_window already split it into features and label, no need to supply batch size because it is already defined in the windowGenerator class
# %%

single_output_window.plot(plot_col='cases_new', model=model)

# %%
