def preprocess_data(data):
    """
    数据预处理函数：为交易数据添加时间相关的特征。
    参数：
        data (DataFrame): 包含交易数据的 Pandas DataFrame(包含 'Time' 列)
    返回：
        DataFrame: 添加了时间特征的 DataFrame
    """
    # 将 'Time' 列的值转换为小时数（0-23），并存储在新列 'Hour' 中
    data['Hour'] = data['Time'] // 3600 % 24

    # 将 'Time' 列的值转换为分钟数（0-59），并存储在新列 'Minute' 中
    data['Minute'] = (data['Time'] // 60) % 60

    # 计算 'Time' 列中相邻行之间的时间差，并存储在新列 'Time_Diff' 中
    # 如果是第一行，时间差为 0（使用 fillna(0) 填充缺失值）
    data['Time_Diff'] = data['Time'].diff().fillna(0)

    # 返回添加了新特征的 DataFrame
    return data
    