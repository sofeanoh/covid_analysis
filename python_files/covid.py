#%%  import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
# %% Define constants
DATASETS_PATH = os.path.join(os.getcwd(), os.pardir, 'datasets')
TRAIN_PATH = os.path.join(DATASETS_PATH, 'cases_malaysia_train.csv')
TEST_PATH = os.path.join(DATASETS_PATH, 'cases_malaysia_test.csv')
# %%Data Loading
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
# %% Data Exploration
train_df.head()
test_df.head()

# %%
