"""
A file for pulling in the data and cleaning it up for the project.
"""

# importing packages
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file: "excel file") -> pd.DataFrame:
    """
    Load in the data for the project
    Returns: A df of the data
    """
    try:
        dataframe = pd.read_excel(file, index_col="Golfer")
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
        pulled_df.replace(-1, 999, inplace=True)
        return pulled_df
    except Exception as e:
        print(f"Error cleaning the data: {e}")


df_pull = load_data("2024_masters_schedules.xlsx")
df_clean = clean_data(df_pull)
# print(df_clean.head())
print(df_clean.shape)
# find and replace all values in the dataframe that are -1 with 999

