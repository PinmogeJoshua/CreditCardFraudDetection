import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test, selected_features):
    """使用 SHAP 分析模型"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[selected_features])
    
    # 新版本LightGBM兼容写法
    if isinstance(shap_values, list):
        # 二分类：shap_values[0]是负类，shap_values[1]是正类
        shap.summary_plot(shap_values[1], X_test[selected_features], plot_type='bar')
    else:
        # 回归或单分类
        shap.summary_plot(shap_values, X_test[selected_features], plot_type='bar')