import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 设置字体为 SimHei（黑体），支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_transaction_amount(data):
    """
    绘制交易金额分布图，展示交易金额的频率分布。

    参数:
        data (DataFrame): 包含交易金额的 Pandas DataFrame，必须包含 'Amount' 列。
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Amount'], bins=50, kde=True, color='green')
    plt.title('交易金额分布')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_correlation_heatmap(data):
    """
    绘制特征相关性热力图，展示数据集中各特征之间的相关性。

    参数:
        data (DataFrame): 包含特征的 Pandas DataFrame。
    """
    plt.figure(figsize=(16, 12))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性热力图')
    plt.show()


def plot_feature_density(data):
    """
    绘制特征密度图，展示每个特征在不同类别（Class = 0 和 Class = 1）下的分布。

    参数:
        data (DataFrame): 包含特征和目标列的 Pandas DataFrame，必须包含 'Class' 列。
    """
    var = data.columns.values
    t0 = data.loc[data['Class'] == 0]
    t1 = data.loc[data['Class'] == 1]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(8, 5, figsize=(20, 28))  # 调整为 8 行 5 列

    for i, feature in enumerate(var):
        # 跳过目标列 'Class'
        if feature == 'Class':
            continue

        # 检查方差是否为 0，避免密度估计错误
        if t0[feature].var() == 0 or t1[feature].var() == 0:
            print(f"特征 {feature} 在某个类别中的方差为 0，跳过密度估计")
            continue
        
        plt.subplot(8, 5, i + 1)
        sns.kdeplot(t0[feature], bw_adjust=0.5, label="Class = 0")
        sns.kdeplot(t1[feature], bw_adjust=0.5, label="Class = 1")
        plt.xlabel(feature, fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()