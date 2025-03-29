from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay
import joblib

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)
    print("Test AUPRC:", auprc)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # DET曲线
    DetCurveDisplay.from_predictions(y_test, y_pred)
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.grid(True)
    plt.show()
    
    # 模型保存
    joblib.dump(model, "lgb_optimized_model.pkl")
    print("模型已保存为 lgb_optimized_model.pkl")
    
