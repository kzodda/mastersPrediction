"""
Models to be implemented
"""

# importing packages
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV


def basic_model(X, y):
    """
    A basic model with default params
    Returns: IDK yet
    """

    y_array = np.array(y, dtype=int)
    model = xgb.XGBClassifier()

    cv_scores = cross_val_score(model, X, y_array, cv=3, scoring='accuracy')

    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Individual fold scores: {cv_scores}")


def hard_coded_tuning_model(X, y):
    """
    A model with some hyperparameters tuned manually
    Args:
        X: Train
        y: Train

    Returns: Nothing, just prints out the results

    """
    y_array = np.array(y, dtype=int)
    model = xgb.XGBClassifier(
        # n_estimators default is 100
        n_estimators=200,
        # learning_rate default is 0.3
        learning_rate=0.1,
        # max_depth default is 6
        max_depth=4
    )

    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Individual fold scores: {scores}")


def tuning_model(X, y):
    """
    A hyper tuning model
    Args:
        X: train
        y: Train

    Returns: Nothing just prints out the results

    """
    y_array = np.array(y, dtype=int)

    # Now we create a parameter grid to search over using a dictionary
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.5, 0.75, 1],
        'colsample_bytree': [0.5, 0.75, 1]
    }

    xgb_model = xgb.XGBClassifier(objective='binary:logistic')

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X, y_array)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    best_cv_scores = cross_val_score(grid_search.best_estimator_, X, y_array, cv=5, scoring='accuracy')
    print(f"Best model CV accuracy: {best_cv_scores.mean():.4f} ± {best_cv_scores.std():.4f}")


def feature_engineering_on_base_model(X, y):
    """
    A function that will do some feature engineering on the base model
    Args:
        X: Train
        y: Train
    Returns: Nothing yet

    """
    y_array = np.array(y, dtype=int)
    model = xgb.XGBClassifier(
        # n_estimators default is 100
        n_estimators=200,
        # learning_rate default is 0.3
        learning_rate=0.1,
        # max_depth default is 6
        max_depth=4
    )

    model.fit(X, y_array)
    feature_importances = model.feature_importances_
    pairs = list(zip(X.columns, feature_importances))
    bad_pairs = [j[0] for i, j in enumerate(pairs) if j[1] <= 0.003]
    # print(f"Bad Cols: {bad_pairs}")
    return bad_pairs


def final_model(X_train, y_train, X_test, y_test):
    """
    This is the final model
    Args:
        X_train: Train
        y_train: Train
        X_test: Test
        y_test: Test

    Returns: Nothing but prints the final scores

    """
    ytrain_array = np.array(y_train, dtype=int)
    ytest_array = np.array(y_test, dtype=int)
    model = xgb.XGBClassifier(
        n_estimators=300,
        subsample=0.5,
        max_depth=4,
        learning_rate=0.1,
        colsample_bytree=1
    )

    model.fit(X_train, ytrain_array)

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == ytest_array)
    accuracy_score = model.score(X_test, ytest_array)
    print(f"Final model accuracy: {accuracy:.2f}")
