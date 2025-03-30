from config import DATA_PATH
from src.data.loader import load_data, inspect_data

from src.data.preprocessing import preprocess_data

from src.features.engineer import create_time_features, scale_features
from src.features.selector import select_features

from src.models.trainer import train_model, save_model
from src.models.evaluator import evaluate_model

from src.explainability.shap_analysis import explain_model

from src.utils.visualizer import plot_precision_recall_curve, plot_feature_importance
from src.utils.explainer import explain_model

from src.visualization.plots import plot_fraud_hours, plot_class_distribution
from src.visualization.advanced_plots import (
    plot_transaction_amount,
    plot_correlation_heatmap,
    plot_feature_density
)

def main():
    # 1. 加载数据
    data = load_data(DATA_PATH)

    # 1.1 检查数据集基本信息和缺失值
    inspect_data(data)
    
    # 1.2 可视化目标列的类别分布
    plot_class_distribution(data, target_column='Class')
    
    # 1.3 可视化交易金额分布
    plot_transaction_amount(data)

    # 2. 数据预处理
    data = preprocess_data(data)

    # 3. 数据可视化
    plot_fraud_hours(data)

    # 4.1 创建时间特征
    data = create_time_features(data)

    # 4.2 标准化数值特征
    # 默认标准化 'Amount' 列, 也可以进一步扩展其他列
    data = scale_features(data, columns_to_scale=['Amount', 'Time_Diff'])
    
    # 4.3 特征相关性热力图
    plot_correlation_heatmap(data)

    # 4.4 特征密度图
    plot_feature_density(data)
    
    # 5. 特征选择
    X_train, X_test, y_train, y_test, selected_features = select_features(data)

    # 6. 模型训练
    model = train_model(X_train, y_train, X_test, y_test)

    # 7.1 模型评估
    # evaluate_model 函数返回 precision, recall 和 auprc
    precision, recall, auprc = evaluate_model(model, X_test, y_test)

    # 7.2 绘制 Precision-Recall 曲线
    plot_precision_recall_curve(precision, recall, auprc)

    # 8.1 模型解释性分析
    explain_model(model, X_test, selected_features)
    
    # 8.2 绘制特征重要性条形图
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(
            feature_importances=model.feature_importances_,
            feature_names=selected_features,
            top_n=10
        )
    else:
        print("模型不支持 feature_importances_ 属性，无法绘制特征重要性图")

    # 9. 保存模型
    save_model(model, "models/lightgbm_model.pkl")

if __name__ == "__main__":
    main()
    