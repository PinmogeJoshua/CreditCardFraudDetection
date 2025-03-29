from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import numpy as np

def tune_hyperparameters(X_train, y_train):
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    
    # === max_depth和num_leaves ===
    param_grid1 = {
        'max_depth': [4, 6, 8, 10],
        'num_leaves': [15, 31, 63, 127]
    }
    grid_search1 = GridSearchCV(
        estimator=model,
        param_grid=param_grid1,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search1.fit(X_train, y_train)
    print("Best parameters found:", grid_search1.best_params_)
    print("Best cross-validation AUC score:", grid_search1.best_score_)

    # === min_data_in_leaf ===
    param_grid2 = {
        'min_child_samples': [18, 19, 20, 21, 22],
        'min_child_weight': [0.001, 0.002]
    }
    grid_search2 = GridSearchCV(
        estimator=model,
        param_grid=param_grid2,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search2.fit(X_train, y_train)
    print("Best parameters found:", grid_search2.best_params_)
    print("Best cross-validation AUC score:", grid_search2.best_score_)

    # === feature_fraction ===
    param_grid3 = {
        'feature_fraction': [0.75, 0.8, 0.85],
        'bagging_fraction': [0.55, 0.6, 0.65]
    }
    grid_search3 = GridSearchCV(
        estimator=model,
        param_grid=param_grid3,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search3.fit(X_train, y_train)
    print("Best parameters found:", grid_search3.best_params_)
    print("Best cross-validation AUC score:", grid_search3.best_score_)

    # === 正则化参数 ===
    param_grid4 = {
        'lambda_l1': [0, 0.1, 0.5],
        'lambda_l2': [0, 0.1, 0.5]
    }
    grid_search4 = GridSearchCV(
        estimator=model,
        param_grid=param_grid4,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search4.fit(X_train, y_train)
    print("Best parameters found:", grid_search4.best_params_)
    print("Best cross-validation AUC score:", grid_search4.best_score_)

    # 返回所有最佳参数
    return {
        **grid_search1.best_params_,
        **grid_search2.best_params_,
        **grid_search3.best_params_,
        **grid_search4.best_params_
    }
    