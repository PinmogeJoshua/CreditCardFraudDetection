from sklearn.model_selection import train_test_split

def split_data(data, target_column='Class', test_size=0.2, random_state=42, stratify=True):
    """
    将数据集划分为训练集和测试集。

    参数:
    - data: 数据集 (DataFrame)
    - target_column: 目标列名称 (默认 'Class')
    - test_size: 测试集比例 (默认 0.2)
    - random_state: 随机种子 (默认 42)
    - stratify: 是否按目标列分层 (默认 True)

    返回:
    - X_train: 训练集特征
    - X_test: 测试集特征
    - y_train: 训练集标签
    - y_test: 测试集标签
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test