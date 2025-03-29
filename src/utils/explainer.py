import shap
import matplotlib.pyplot as plt
import pandas as pd

def shap_analysis(model, X_test):
    """对应notebook第6章可解释性分析"""
    # 严格保持原始notebook代码（包括打印语句）
    print("\nSHAP分析部分：")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 保留原始的类型检查代码
    print(f"shap_values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"shap_values[0] shape: {shap_values[0].shape}")
        print(f"shap_values[1] shape: {shap_values[1].shape}")
    
    # 绘制SHAP图（保持原始参数）
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, plot_type='bar', max_display=15)
    else:
        shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=15)
    
    plt.tight_layout()
    plt.show()