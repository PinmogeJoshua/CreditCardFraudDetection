def preprocess_data(data):
    """数据预处理：添加时间特征"""
    data['Hour'] = data['Time'] // 3600 % 24
    data['Minute'] = (data['Time'] // 60) % 60
    data['Time_Diff'] = data['Time'].diff().fillna(0)
    return data
    