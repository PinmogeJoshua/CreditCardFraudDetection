from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

def select_features(data):
    """特征选择"""
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        threshold='median'
    ).fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"最终选择特征：{selected_features}")
    return X_train, X_test, y_train, y_test, selected_features