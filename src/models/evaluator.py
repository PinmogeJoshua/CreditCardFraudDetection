from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay
import joblib

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    
    DetCurveDisplay.from_predictions(y_test, y_pred)
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.show()
    
    joblib.dump(model, "lgb_model.pkl")
    return auprc
