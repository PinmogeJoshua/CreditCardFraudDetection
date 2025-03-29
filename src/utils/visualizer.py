import matplotlib.pyplot as plt
import seaborn as sns

def plot_fraud_time_pattern(data):
    plt.figure(figsize=(12, 4))
    fraud_hours = data[data['Class']==1]['Hour'].value_counts().sort_index()
    sns.barplot(x=fraud_hours.index, y=fraud_hours.values, palette='Reds')
    plt.title('欺诈交易的小时分布')
    plt.show()

def plot_correlation(data):
    corr_with_class = data.corr()['Class'].drop('Class').sort_values(ascending=False)  
    plt.figure(figsize=(8, 10))
    sns.barplot(y=corr_with_class.index, x=corr_with_class.values, palette='coolwarm')
    plt.title('特征与Class的相关性排序')
    plt.show()
    