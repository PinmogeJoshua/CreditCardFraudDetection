import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test, selected_features):
    """使用 SHAP 分析模型"""
    # 创建解释器
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值（新版本LightGBM的输出格式不同）
    shap_values = explainer.shap_values(X_test[selected_features])
    
    # 可视化 - 显示两类SHAP值
    if isinstance(shap_values, list):
        # 二分类任务会返回两个矩阵（分别对应两类）
        shap.summary_plot(shap_values[1], X_test[selected_features], plot_type='bar')
    else:
        # 单矩阵输出（回归或单类分类）
        shap.summary_plot(shap_values, X_test[selected_features], plot_type='bar')