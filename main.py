"""
The main file for the project. This file will be used to run the project.
"""
# importing packages and modules
import config
from sklearn.model_selection import train_test_split
import logging
from models import (basic_model, hard_coded_tuning_model,
                    tuning_model, feature_engineering_on_base_model, final_model)
from data_utils import data_operator


def main():
    df = data_operator()
    # print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], df.iloc[:, -1],
                                                        test_size=0.05,
                                                        random_state=42, shuffle=True, stratify=None)
    # Creating a basic model with default vals
    # basic_model(X_train, y_train)

    # Trying hard coded tuning model
    # hard_coded_tuning_model(X_train, y_train)

    # Trying the hyper tuning model
    # tuning_model(X_train, y_train)

    # Now I need to do some feature engineering and see which cols to drop
    cols_to_drop = feature_engineering_on_base_model(X_train, y_train)
    # Now I need to drop them from both the TRAIN and TEST sets
    X_train.drop(columns=cols_to_drop, inplace=True)
    X_test.drop(columns=cols_to_drop, inplace=True)

    # Now I need to run the model again
    # basic_model(X_train, y_train)

    # Now let's Hypertune the model again
    # tuning_model(X_train, y_train)
    final_model(X_train, y_train, X_test, y_test)

    return None


main()
