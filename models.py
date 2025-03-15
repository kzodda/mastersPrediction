"""
Models to be implemented
"""

# importing packages
import xgboost as xgb
from sklearn.model_selection import cross_val_score


def basic_model(X, y):
    """
    A basic model with default params
    Returns: Idk yet
    """
    model = xgb.XGBClassifier()

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Individual fold scores: {cv_scores}")

