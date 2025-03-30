from sklearn.metrics import precision_recall_curve, auc

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能，并返回 Precision-Recall 曲线相关数据。

    参数:
        model: 已训练的模型。
        X_test: 测试集特征 (DataFrame)。
        y_test: 测试集标签 (Series)。

    返回:
        precision: 精确率数组。
        recall: 召回率数组。
        auprc: Precision-Recall 曲线下的面积 (AUPRC)。
    """
    # 使用模型预测测试集
    y_pred = model.predict(X_test)

    # 计算 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # 计算 AUPRC
    auprc = auc(recall, precision)

    # 打印 AUPRC
    print("Test AUPRC:", auprc)

    # 返回 Precision, Recall 和 AUPRC
    return precision, recall, auprc
