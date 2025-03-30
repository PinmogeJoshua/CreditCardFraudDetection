from config import DATA_PATH
from src.data.loader import load_data, inspect_data
from src.data.preprocessing import preprocess_data
from src.features.engineer import create_time_features, scale_features
from src.features.selector import select_features
from src.models.trainer import train_model, save_model
from src.models.evaluator import evaluate_model
from src.explainability.shap_analysis import explain_model
from src.visualization.plots import plot_fraud_hours, plot_class_distribution

def main():
    # 1. 加载数据
    data = load_data(DATA_PATH)

    # 1.1 检查数据集基本信息和缺失值
    inspect_data(data)
    
    # 1.2 可视化目标列的类别分布
    plot_class_distribution(data, target_column='Class')

    # 2. 数据预处理
    data = preprocess_data(data)

    # 3. 数据可视化
    plot_fraud_hours(data)

    # 4. 特征工程
    data = create_time_features(data)
    data = scale_features(data)

    # 5. 特征选择
    X_train, X_test, y_train, y_test, selected_features = select_features(data)

    # 6. 模型训练
    model = train_model(X_train, y_train, X_test, y_test)

    # 7. 模型评估
    evaluate_model(model, X_test, y_test)

    # 8. 模型解释性分析
    explain_model(model, X_test, selected_features)

    # 9. 保存模型
    save_model(model, "models/lightgbm_model.pkl")

if __name__ == "__main__":
    main()
    