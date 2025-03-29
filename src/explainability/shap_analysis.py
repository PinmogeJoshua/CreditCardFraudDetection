import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test, selected_features):
    """使用 SHAP 分析模型"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[selected_features])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test[selected_features], plot_type='bar', max_display=15)