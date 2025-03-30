import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot

def plot_fraud_hours(data):
    """绘制欺诈交易的小时分布"""
    plt.figure(figsize=(12, 4))
    fraud_hours = data[data['Class'] == 1]['Hour'].value_counts().sort_index()
    sns.barplot(x=fraud_hours.index, y=fraud_hours.values, palette='Reds')
    plt.title('欺诈交易的小时分布')
    plt.show()
    
def plot_class_distribution(data, target_column='Class'):
    """
    可视化目标列的类别分布，呈现数据不平衡情况

    参数:
        data (DataFrame): 包含数据的 Pandas DataFrame
        target_column (str): 目标列名称，默认为 'Class'

    返回:
        None
    """
    # 统计目标列中每个类别的数量
    temp = data[target_column].value_counts()
    df = pd.DataFrame({'Class': temp.index, 'values': temp.values})

    # 创建柱状图
    trace = go.Bar(
        x=df['Class'],
        y=df['values'],
        name=f"{target_column} - 数据不平衡 (Not fraud = 0, Fraud = 1)",
        marker=dict(color="Red"),
        text=df['values']
    )
    data = [trace]
    layout = dict(
        title=f'{target_column} - 数据不平衡 (Not fraud = 0, Fraud = 1)',
        xaxis=dict(title='Class', showticklabels=True),
        yaxis=dict(title='Number of transactions'),
        hovermode='closest',
        width=600
    )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='class_distribution')
    