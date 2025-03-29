# 大数据管理方法与应用个人作业

## 项目简介
本项目旨在通过机器学习模型检测信用卡交易中的欺诈行为。数据集来源于公开的信用卡交易数据，包含匿名特征和交易标签（Class 列，1 表示欺诈，0 表示正常交易）。项目实现了从数据加载、预处理、特征工程到模型训练、评估和解释的完整流程
## 目录结构
CreditCardFraudDetection/
│
├── main.py                     # 主程序入口
├── config.py                   # 配置文件
├── requirements.txt            # 项目依赖文件
├── README.md                   # 项目说明文档
├── setup.py                    # 项目打包文件（可选）
│
├── data/                       # 数据文件夹
│   ├── creditcard.csv          # 原始数据文件
│   └── processed/              # 处理后的数据文件夹
│
├── models/                     # 模型文件夹
│   ├── lightgbm_model.pkl      # 保存的模型文件
│   └── __init__.py             # 包初始化文件
│
├── logs/                       # 日志文件夹
│   └── training.log            # 训练日志
│
├── notebooks/                  # Jupyter Notebook 文件夹
│   └── eda.ipynb               # 数据探索分析
│   └── experiments.ipynb       # 实验记录
│
├── src/                        # 源代码文件夹
│   ├── data/                   # 数据处理模块
│   │   ├── loader.py           # 数据加载模块
│   │   ├── preprocessing.py    # 数据预处理模块
│   │   └── splitter.py         # 数据集划分模块
│   │
│   ├── features/               # 特征工程模块
│   │   ├── engineer.py         # 特征生成模块
│   │   └── selector.py         # 特征选择模块
│   │
│   ├── models/                 # 模型相关模块
│   │   ├── trainer.py          # 模型训练模块
│   │   └── evaluator.py        # 模型评估模块
│   │
│   ├── visualization/          # 可视化模块
│   │   └── plots.py            # 绘图代码
│   │
│   ├── explainability/         # 模型解释性分析模块
│   │   └── shap_analysis.py    # SHAP 分析代码
│   │
│   └── utils/                  # 工具函数模块
│       ├── __init__.py         # 包初始化文件
│       ├── explainer.py        # 模型解释工具
│       └── visualizer.py       # 可视化工具
│
└── tests/                      # 测试模块
    ├── test_data_loader.py     # 测试数据加载模块
    ├── test_feature_engineer.py# 测试特征工程模块
    └── test_model_trainer.py   # 测试模型训练模块
## 功能模块

### 数据处理模块
loader.py: 加载数据集。
preprocessing.py: 数据预处理（如时间特征生成）。
splitter.py: 数据集划分（训练集和测试集）。
### 特征工程模块
engineer.py: 特征生成（如时间特征的正余弦转换）。
selector.py: 特征选择（基于逻辑回归的 L1 正则化）。
### 模型模块
trainer.py: 模型训练（使用 LightGBM 和 SMOTE 处理不平衡数据）。
evaluator.py: 模型评估（计算 Precision-Recall 曲线和 AUPRC）。
### 可视化模块
plots.py: 绘制欺诈交易分布、Precision-Recall 曲线等。
### 模型解释性分析模块
shap_analysis.py: 使用 SHAP 分析模型的特征重要性。
### 工具模块
explainer.py: 模型解释工具。
visualizer.py: 通用可视化工具。
### 测试模块
test_data_loader.py: 测试数据加载模块。
test_feature_engineer.py: 测试特征工程模块。
test_model_trainer.py: 测试模型训练模块。

## 安装步骤
1. 克隆项目到本地：
git clone https://github.com/your-repo/CreditCardFraudDetection.git
cd CreditCardFraudDetection

2. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt

3. 确保数据文件creditcard.csv位于data文件夹中

## 使用方法
1. 运行主程序
python main.py

2. 查看日志文件： 日志文件位于 training.log，记录了模型训练和评估的详细信息。

3. 查看模型文件： 训练完成后，模型会保存到 models/lightgbm_model.pkl。

## 测试
运行以下命令执行单元测试：
python -m unittest discover tests

## 项目依赖
项目依赖的主要 Python 包包括：

pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
shap
imbalanced-learn
完整依赖请参考 requirements.txt 文件。

## 数据集说明
数据集文件：data/creditcard.csv
数据集包含匿名特征 V1 到 V28，以及交易金额 Amount 和时间 Time。
标签列 Class：1 表示欺诈交易，0 表示正常交易。

## 主要功能
1. 数据加载与预处理：
    加载数据集并生成时间特征
    使用 SMOTE 处理数据不平衡问题
2. 特征工程：
    生成时间特征的正余弦转换
    使用逻辑回归的 L1 正则化进行特征选择
3. 模型训练与评估：
    使用 LightGBM 训练模型
    评估模型的 Precision-Recall 曲线和 AUPRC
4. 可视化与解释：
    绘制欺诈交易分布图
    使用 SHAP 分析模型的特征重要性

## 贡献
如果你对本项目有任何建议或改进，欢迎提交 Issue 或 Pull Request！

## 许可证
本项目遵循 MIT License。