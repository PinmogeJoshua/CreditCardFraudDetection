#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
信用卡欺诈检测主流程
"""

from src.data.loader import load_data
from src.data.preprocessor import process_time_features, scale_features, plot_distributions
from src.utils.visualizer import plot_fraud_time_pattern, plot_correlation
from src.features.selector import select_features
from src.features.engineer import add_cyclic_features
from src.models.trainer import handle_imbalance, train_lightgbm  
from src.models.tuner import tune_hyperparameters  
from src.models.evaluator import evaluate_model
from src.utils.explainer import shap_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

def main():
    # ==================== 1. 数据加载 ====================
    print("\n" + "="*20 + " 1. 数据加载 " + "="*20)
    data = load_data()
    
    # ==================== 2. 数据预处理 ====================
    print("\n" + "="*20 + " 2. 数据预处理 " + "="*20)
    data = process_time_features(data)
    
    # 2.2 数据分布检查
    print("\n数据分布直方图：")
    plot_distributions(data)
    
    # ==================== 3. 数据可视化 ====================
    print("\n" + "="*20 + " 3. 数据可视化 " + "="*20)
    plot_fraud_time_pattern(data)
    plot_correlation(data)
    
    # ==================== 4. 特征工程 ====================
    print("\n" + "="*20 + " 4. 特征工程 " + "="*20)
    # 4.1 特征缩放
    data = scale_features(data)
    data = add_cyclic_features(data)
    
    # 4.2 时间序列分割
    data_sorted = data.sort_values('Time')
    X = data_sorted.drop('Class', axis=1)
    y = data_sorted['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n训练集类别分布:")
    print(y_train.value_counts(normalize=True))
    print("\n测试集类别分布:")
    print(y_test.value_counts(normalize=True))
    
    # 4.3 特征选择
    selected_features = select_features(X_train, y_train)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"\n最终选择特征：{selected_features}")
    
    # 4.4 处理不平衡数据
    class_counts = y_train.value_counts()
    class_ratios = y_train.value_counts(normalize=True)
    print("\n类别计数:\n", class_counts)
    print("\n类别比例:\n", class_ratios)
    
    X_train_resampled, y_train_resampled, class_weight = handle_imbalance(X_train, y_train)
    
    # ==================== 5. 模型训练与评估 ====================
    print("\n" + "="*20 + " 5. 模型训练与评估 " + "="*20)
    
    # 5.3 超参数调优
    print("\n====== 超参数调优开始 ======")
    tune_hyperparameters(X_train_resampled, y_train_resampled)
    
    # 5.1 训练模型
    train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    model = train_lightgbm(X_train_resampled, y_train_resampled, X_test, y_test, class_weight)
    
    # 5.2 模型评估
    evaluate_model(model, X_test, y_test)
    
    # ==================== 6. 可解释性分析 ====================
    print("\n" + "="*20 + " 6. 可解释性分析 " + "="*20)
    shap_analysis(model, X_test)
    print("\n" + "="*20 + " 流程执行完成 " + "="*20)

if __name__ == "__main__":
    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
    