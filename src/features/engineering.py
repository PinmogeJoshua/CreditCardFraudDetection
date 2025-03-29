import numpy as np
from sklearn.preprocessing import StandardScaler

def create_time_features(data):
    """创建时间特征"""
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    return data

def scale_features(data):
    """对特征进行标准化"""
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])
    return data