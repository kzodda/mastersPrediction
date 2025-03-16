"""
A file for pulling in the data and cleaning it up for the project.
"""

# importing packages
import config
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
from pathlib import Path


def load_data(file: "excel file") -> pd.DataFrame:
    """
    Load in the data for the project
    Returns: A df of the data
    """
    script_dir = Path(__file__).parent.absolute()
    file_path = script_dir / file

    try:
        dataframe = pd.read_excel(file_path, index_col="Golfer")
        return dataframe
    except Exception as e:
        print(f"Error loading the data: {e}")
        return None


def clean_data(pulled_df: pd.DataFrame) -> pd.DataFrame:
    """
    A dataframe cleaning function for the project
    Args:
        pulled_df: The dataframe that was pulled

    Returns: a Cleaned dataframe

    """
    try:
        pulled_df = pulled_df.replace({
            -1: 999,
            "Yes": 1,
            "No": 0
        }).astype(int)
        # print(pulled_df.head())
        # print(pulled_df.iloc[0, 0])
        return pulled_df
    except Exception as e:
        print(f"Error cleaning the data: {e}")


def data_operator():
    """
    A function that will run the data operations
    Returns: None

    """
    print("Starting the data operations")
    df_pull = load_data(r"2024_masters_schedules.xlsx")
    # print(df_pull.head())
    df_clean = clean_data(df_pull)
    # print(df_clean.columns)
    # print(df_clean.iloc[0, :])
    # print(df_clean.head())
    # print(df_clean.iloc[:, -1].isnull().sum())
    return df_clean
