import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def add_cyclic_features(data):
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    return data

def scale_features(data):
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])
    
    scaler_minmax = MinMaxScaler()
    data_scaled = scaler_minmax.fit_transform(data)  # 原样保留未使用的变量
    
    return data
