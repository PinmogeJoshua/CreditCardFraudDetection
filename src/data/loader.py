import pandas as pd

def load_data(filepath):
    """
    加载数据集
    
    参数:
        filepath: 文件路径

    返回:  
        DataFrame: 数据集
    """
    data = pd.read_csv(filepath)
    print("数据维度:", data.shape)
    return data

def inspect_data(data):
    """
    检查数据集的基本信息和缺失值情况。

    参数:
        data (DataFrame): 数据集

    返回:
        None
    """
    print("数据集基本信息:")
    print(data.info())
    print("\n数据集缺失值统计:")
    print(data.isnull().sum())
    