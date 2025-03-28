# Basic libraries
import pandas as pd
import numpy as np

# Machine Learning libraries
from sklearn.model_selection import KFold

def target_encode_cv_smooth(df, cat_col, target_col, k=5, n_splits=5, random_state=42) -> pd.Series:
    """
    Cross-validated target encoding with Bayesian smoothing.
    
    Parameters:
        df (pd.DataFrame): Your dataframe.
        cat_col (str): Categorical column to encode.
        target_col (str): Target variable (e.g., amount_paid_usd).
        k (int): Smoothing factor.
        n_splits (int): Number of KFold splits.
        random_state (int): Random seed.

    Returns:
        pd.Series: Encoded column with smoothed mean target per category.
    """
    df = df.copy()
    global_mean = df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    encoded = pd.Series(index=df.index, dtype=np.float64)

    for train_idx, val_idx in kf.split(df):
        train, val = df.iloc[train_idx], df.iloc[val_idx]

        agg = train.groupby(cat_col)[target_col].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']

        smooth = (counts * means + k * global_mean) / (counts + k)

        val_encoded = val[cat_col].map(smooth)
        encoded.iloc[val_idx] = val_encoded.fillna(global_mean)

    return encoded