import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def download_kaggle_dataset(dataset_slug: str, file_name: str, **pandas_kwargs) -> pd.DataFrame:
    """
    Downloads a CSV file from a Kaggle dataset and loads it as a pandas DataFrame.

    Parameters:
    - dataset_slug (str): The full Kaggle dataset identifier, e.g., "mathurinache/1000000-bandcamp-sales"
    - file_path (str): Path to the CSV file within the Kaggle dataset.
    - pandas_kwargs: Additional kwargs to pass to pandas read_csv.

    Returns:
    - pd.DataFrame
    """
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_slug,
        file_name,
        pandas_kwargs=pandas_kwargs
    )
    return df