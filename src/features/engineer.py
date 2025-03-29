import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def add_cyclic_features(data):
    """对应notebook中4.1节的特征工程代码"""
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    return data

def scale_features(data):
    """对应notebook中4.1节的特征缩放"""
    # 保留原始notebook中的全部代码（包括注释）
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])
    
    # 保留有问题的MinMaxScaler代码（原notebook中的data_scaled未使用）
    scaler_minmax = MinMaxScaler()
    data_scaled = scaler_minmax.fit_transform(data)  # 原样保留未使用的变量
    
    return data