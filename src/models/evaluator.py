from sklearn.metrics import precision_recall_curve, auc

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)
    print("Test AUPRC:", auprc)
