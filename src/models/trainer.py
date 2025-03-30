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
    # 动态计算正负样本比例（SMOTE前原始比例）
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    
    # 复制并更新参数（移除is_unbalance）
    params = BASE_PARAMS.copy()
    params.update({
        'scale_pos_weight': scale_pos_weight,
        'is_unbalance': False  # 显式禁用
    })
    print(f"正负样本比例: 1:{scale_pos_weight:.1f}")

    # 应用SMOTE（仅训练集）
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # 训练模型（注意eval_set用原始测试集）
    model = lgb.train(
        params,
        train_set=lgb.Dataset(X_resampled, y_resampled),
        valid_sets=[lgb.Dataset(X_test, y_test)],  # 用原始测试集评估
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    return model

def save_model(model, filepath):
    """保存模型"""
    import joblib
    joblib.dump(model, filepath)
    print(f"模型已保存为 {filepath}")
