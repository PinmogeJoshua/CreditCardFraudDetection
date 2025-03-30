from sklearn.model_selection import train_test_split

def split_data(data, target_column='Class', test_size=0.2, random_state=42, stratify=True):
    """
    将数据集划分为训练集和测试集

    参数:
    - data: 数据集 (DataFrame)
    - target_column: 目标列名称, 表示分类任务标签列
    - test_size: 测试集比例, 表示测试集占总数据集的比例
    - random_state: 随机种子, 用于保证划分的可重复性
    - stratify: 是否按目标列分层 (默认 True), 如果为True, 则按照目标列的分布划分数据

    返回:
    - X_train: 训练集特征 (DataFrame)
    - X_test: 测试集特征 (DataFrame)
    - y_train: 训练集标签 (Series)
    - y_test: 测试集标签 (Series)
    """
    # 将目标列从数据集中移除，得到特征数据 X
    X = data.drop(columns=[target_column])
    # 提取目标列作为标签数据 y
    y = data[target_column]

    # 如果 stratify 为 True，则按目标列 y 的分布进行分层抽样
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # 如果 stratify 为 False，则随机划分数据集，不考虑目标列的分布
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # 打印训练集和测试集的大小，方便调试和验证
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 返回训练集和测试集的特征及标签
    return X_train, X_test, y_train, y_test
