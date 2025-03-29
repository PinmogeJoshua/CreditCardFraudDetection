from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_features(X_train, y_train):
    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        threshold='median'
    ).fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features