import matplotlib.pyplot as plt
import seaborn as sns

def plot_fraud_hours(data):
    """绘制欺诈交易的小时分布"""
    plt.figure(figsize=(12, 4))
    fraud_hours = data[data['Class'] == 1]['Hour'].value_counts().sort_index()
    sns.barplot(x=fraud_hours.index, y=fraud_hours.values, palette='Reds')
    plt.title('欺诈交易的小时分布')
    plt.show()