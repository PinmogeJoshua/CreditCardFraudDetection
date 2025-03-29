from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import TimeSeriesSplit

def handle_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    class_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    return X_resampled, y_resampled, class_weight

def train_model(X_train, y_train, X_test, y_test, class_weight):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'dart',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'scale_pos_weight': class_weight
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=500,
        callbacks=[early_stopping(stopping_rounds=50)]
    )
    return model