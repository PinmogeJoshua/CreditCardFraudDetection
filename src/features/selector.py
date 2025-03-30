from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

def select_features(data):
    """
    特征选择函数：通过 L1 正则化的逻辑回归模型选择重要特征。

    参数:
        data (DataFrame): 包含特征和目标列的 Pandas DataFrame，目标列名为 'Class'。

    返回:
        X_train (DataFrame): 训练集的特征数据，仅包含被选择的特征。
        X_test (DataFrame): 测试集的特征数据，仅包含被选择的特征。
        y_train (Series): 训练集的目标数据。
        y_test (Series): 测试集的目标数据。
        selected_features (list): 被选择的重要特征名称列表。
    """
    # 1. 分离特征和目标列
    X = data.drop('Class', axis=1)  # 特征数据
    y = data['Class']  # 目标数据

    # 2. 划分训练集和测试集，按目标列分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 使用 L1 正则化的逻辑回归模型进行特征选择
    # SelectFromModel 会根据模型的系数选择重要特征
    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        threshold='median'  # 选择系数绝对值大于中位数的特征
    ).fit(X_train, y_train)

    # 4. 获取被选择的特征名称
    selected_features = X_train.columns[selector.get_support()].tolist()

    # 5. 根据选择的特征重新构建训练集和测试集
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # 打印最终选择的特征
    print(f"最终选择特征：{selected_features}")

    # 6. 返回处理后的数据和选择的特征列表
    return X_train, X_test, y_train, y_test, selected_features
