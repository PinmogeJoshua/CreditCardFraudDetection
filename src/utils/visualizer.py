import matplotlib.pyplot as plt
import seaborn as sns

def plot_fraud_hours(data):
    """
    绘制欺诈交易的小时分布。
    
    参数:
    - data: 包含交易数据的 DataFrame，必须包含 'Class' 和 'Hour' 列。
    """
    plt.figure(figsize=(12, 4))
    fraud_hours = data[data['Class'] == 1]['Hour'].value_counts().sort_index()
    sns.barplot(x=fraud_hours.index, y=fraud_hours.values, palette='Reds')
    plt.title('欺诈交易的小时分布')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Fraudulent Transactions')
    plt.show()

def plot_precision_recall_curve(precision, recall, auprc):
    """
    绘制 Precision-Recall 曲线。
    
    参数:
    - precision: 精确率数组。
    - recall: 召回率数组。
    - auprc: Precision-Recall 曲线下的面积 (AUPRC)。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_feature_importance(feature_importances, feature_names, top_n=10):
    """
    绘制特征重要性条形图。
    
    参数:
    - feature_importances: 特征重要性数组。
    - feature_names: 特征名称列表。
    - top_n: 显示的最重要特征数量 (默认 10)。
    """
    sorted_idx = feature_importances.argsort()[-top_n:][::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=feature_importances[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        palette="Blues_r"
    )
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()