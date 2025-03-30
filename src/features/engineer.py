import numpy as np
from sklearn.preprocessing import StandardScaler

def create_time_features(data):
    """
    创建时间特征：将小时特征转换为周期性特征（正弦和余弦）。

    参数:
        data (DataFrame): 包含 'Hour' 列的 Pandas DataFrame。

    返回:
        DataFrame: 添加了 'Hour_sin' 和 'Hour_cos' 列的 DataFrame。
    """
    # 检查 'Hour' 列是否存在
    if 'Hour' not in data.columns:
        raise ValueError("数据集中缺少 'Hour' 列，无法创建时间特征。")

    # 将 'Hour' 列转换为周期性特征，使用正弦和余弦表示时间的周期性
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    return data

def scale_features(data, columns_to_scale=None):
    """
    对指定的数值特征进行标准化处理。

    参数:
        data (DataFrame): 包含需要标准化的数值列的 Pandas DataFrame。
        columns_to_scale (list): 需要标准化的列名列表。如果为 None，默认只标准化 'Amount' 列。

    返回:
        DataFrame: 对指定列进行标准化后的 DataFrame。
    """
    # 如果未指定列，默认标准化 'Amount' 列
    if columns_to_scale is None:
        columns_to_scale = ['Amount']

    # 检查指定的列是否存在于数据集中
    for col in columns_to_scale:
        if col not in data.columns:
            raise ValueError(f"数据集中缺少 '{col}' 列，无法进行标准化。")

    # 初始化标准化器
    scaler = StandardScaler()

    # 对指定列进行标准化
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data
