import pandas as pd

def load_data(filepath):
    """加载数据集"""
    data = pd.read_csv(filepath)
    print("数据维度:", data.shape)
    return data
