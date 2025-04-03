# 大数据管理方法与应用个人作业：基于集成学习的信用卡欺诈检测系统

## 项目概述

本项目旨在构建一个高效的信用卡欺诈检测系统，通过多种机器学习模型和先进的集成学习策略，有效识别信用卡交易中的欺诈行为。系统特别关注严重不平衡数据下的分类问题，提供了一套完整的解决方案，从数据预处理、特征工程到模型训练和评估。

## 主要特点

- **高级特征工程**：创建时间相关特征，捕捉交易时间模式
- **不平衡数据处理**：结合SMOTE过采样和类别权重调整
- **多模型集成**：使用Stacking、加权平均等集成学习方法
- **全面评估体系**：采用AUPRC、F1分数等适合不平衡数据的评估指标
- **可解释性分析**：提供特征重要性分析，增强模型透明度

## 项目结构

```
CreditCardFraudDetection/
│
├── data/                      # 数据目录
│   └── creditcard.csv         # 原始数据集
│
├── notebooks/                 # Jupyter笔记本
│   ├── EDA.ipynb              # 探索性数据分析
│   ├── preprocessing.ipynb    # 数据预处理和特征工程
│   ├── feature_selection.ipynb# 特征选择
│   ├── base_model.ipynb       # 基础模型实现
│   └── stacking.ipynb         # 集成学习实现
│
├── outputs/                   # 输出目录
│   ├── datasets/              # 处理后的数据集
│   ├── features/              # 特征相关数据
│   ├── models/                # 训练好的模型
│   └── results/               # 实验结果
│
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   ├── features/              # 特征工程模块
│   ├── models/                # 模型实现模块
│   └── utils/                 # 工具函数
│
├── requirements.txt           # 项目依赖
├── setup.py                   # 安装脚本
└── README.md                  # 项目说明
```

## 主要算法

- **基础模型**
  - LightGBM
  - XGBoost
  - 随机森林
  
- **集成方法**
  - Stacking集成
  - 加权平均集成
  - 特征增强型Stacking

## 性能指标

模型在测试集上的主要性能指标:

| 模型 | AUPRC | F1分数 | 召回率 | 精确率 |
|------|-------|--------|--------|--------|
| Stacking | 0.8580 | 0.8556 | 0.7857 | 0.9390 |

## 使用指南

### 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/CreditCardFraudDetection.git
cd CreditCardFraudDetection

# 创建环境并安装依赖
conda create -n ccfd python=3.9
conda activate ccfd
pip install -r requirements.txt
```

### 运行流程

1. 数据准备与探索
```bash
jupyter notebook notebooks/EDA.ipynb
```

2. 数据预处理与特征工程
```bash
jupyter notebook notebooks/preprocessing.ipynb
```

3. 模型训练与评估
```bash
jupyter notebook notebooks/base_model.ipynb
jupyter notebook notebooks/stacking.ipynb
```

## 数据来源

本项目使用的信用卡交易数据集包含284,807条交易记录，其中欺诈交易占比约0.17%。由于数据隐私原因，大部分特征已通过PCA转换为匿名特征(V1-V28)。

## 贡献指南

欢迎通过以下方式为项目做出贡献:
1. 提交Issue报告bug或提出新功能
2. 提交Pull Request改进代码
3. 优化模型性能或添加新的算法

## 参考文献

- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In KDD.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. In NIPS.
- Chawla, N. V., et al. (2002). SMOTE: synthetic minority over-sampling technique. JAIR.
- Wolpert, D. H. (1992). Stacked generalization. Neural networks, 5(2), 241-259.

## 许可证

MIT

## 联系方式

若有任何问题，请联系 stellewang0417@qq.com
