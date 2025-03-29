import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                           average_precision_score, confusion_matrix)
import lightgbm as lgb
import shap

def load_data():
    data = pd.read_csv('creditcard.csv')
    print("数据维度:", data.shape)
    print(data.head())
    return data
