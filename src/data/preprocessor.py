import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_time_features(data):
    data['Hour'] = data['Time'] // 3600 % 24
    data['Minute'] = (data['Time'] // 60) % 60
    data['Time_Diff'] = data['Time'].diff().fillna(0)
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    return data

def scale_features(data):
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])
    scaler_minmax = MinMaxScaler()
    data_scaled = scaler_minmax.fit_transform(data)
    return data

def plot_distributions(data):
    num_features = len(data.columns)
    cols = 5
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    
    for i, column in enumerate(data.columns):
        if column != 'Class':
            ax = axes[i]
            data[column].hist(bins=30, alpha=0.7, ax=ax)
            ax.set_title(column)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
    
    for i in range(len(data.columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    