import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, auc
from .constants import BASE_PARAMS

def auprc_metric(y_true, y_pred):
    """
    自定义 AUPRC 评估函数，用于 LightGBM。
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率

    返回:
        (str, float, bool): 评估指标名称、AUPRC 值、是否越高越好
    """
    # 处理 y_true 和 y_pred 的不同输入情况
    y_true = np.array(y_true).astype(int)
    
    # 如果 y_pred 是 Dataset 对象（在 LightGBM 内部评估时会发生）
    if hasattr(y_pred, '__class__') and y_pred.__class__.__name__ == 'Dataset':
        # 使用当前模型的预测
        y_pred = y_pred.get_label()
    
    # 确保 y_pred 是 numpy 数组
    y_pred = np.array(y_pred)
    
    # 处理二维数组的情况
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred[:, 1]  # 取正类的概率

    # 计算 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    # 计算 AUPRC
    auprc = auc(recall, precision)

    return 'auprc', auprc, True

def train_model(X_train, y_train, X_test, y_test):
    """训练模型"""
    # 计算原始数据的正负样本比例
    neg_pos_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"原始数据负/正样本比例: {neg_pos_ratio:.2f}:1")
    
    # 使用 SMOTE 进行过采样, 解决数据不平衡问题
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 计算SMOTE后的正负样本比例
    smote_neg_pos_ratio = len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])
    print(f"SMOTE后负/正样本比例: {smote_neg_pos_ratio:.2f}:1")

    # 创建 LightGBM 数据集
    train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
    test_data = lgb.Dataset(X_test, label=y_test)

    # 复制基础参数并调整
    params = BASE_PARAMS.copy()
    
    # 移除is_unbalance, 避免冲突
    if 'is_unbalance' in params:
        params.pop('is_unbalance')
    
    # 训练 LightGBM 模型, 使用自定义 AUPRC 评估函数
    # 使用 BASE_PARAMS 作为参数
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50), # 使用回调函数实现早停
            lgb.log_evaluation(period=50)   # 使用回调函数实现日志输出
            ], 
        feval=auprc_metric
    )
    return model

def save_model(model, filepath):
    """保存模型"""
    import joblib
    joblib.dump(model, filepath)
    print(f"模型已保存为 {filepath}")
