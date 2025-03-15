"""
The main file for the project. This file will be used to run the project.
"""
# importing packages and modules
from sklearn.model_selection import train_test_split
import logging
from models import basic_model
from data_utils import df_clean


def main():
    X_train, X_test, y_train, y_test = train_test_split(df_clean.iloc[:, 0:-1], df_clean.iloc[:, -1], test_size=0.05,
                                                        random_state=42, shuffle=True, stratify=None)
    basic_model(X_train, y_train)
    return None


main()
