from .constants import BASE_PARAMS
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def handle_imbalance(X, y, test_size=0.2, random_state=42):
    """
    使用 SMOTE 处理数据不平衡问题，并划分训练集和测试集。

    参数:
    - X: 特征数据 (DataFrame 或 ndarray)
    - y: 标签数据 (Series 或 ndarray)
    - test_size: 测试集比例 (默认 0.2)
    - random_state: 随机种子 (默认 42)

    返回:
    - X_train: 过采样后的训练特征数据
    - X_test: 测试特征数据
    - y_train: 过采样后的训练标签数据
    - y_test: 测试标签数据
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test

def auprc_eval(preds, train_data):
    labels = train_data.get_label()
    score = average_precision_score(labels, preds)
    return 'auprc', score, True

def train_lightgbm(X_train, y_train, X_test, y_test, class_weight):
    params = BASE_PARAMS.copy()
    
    params.update({
        'boosting_type': 'dart',  # 后续覆盖的参数
        'time_series': True,      # 添加的特殊参数
        'scale_pos_weight': class_weight,
        'is_unbalance': True
    })
    
    # 数据集构建方式
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    # 训练调用（包括所有参数）
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        num_boost_round=500,
        callbacks=[early_stopping(stopping_rounds=50)],
        feval=auprc_eval  # 自定义评估函数
    )
    return model