"""Loading and merging the raw Walmart competition CSVs.

Files expected in `data_dir`:
    train.csv, test.csv, features.csv, stores.csv, sampleSubmission.csv
"""
from pathlib import Path
import pandas as pd


def load_raw(data_dir="data"):
    """Read the five raw CSVs.

    Returns (train, test, features, stores, sample).
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["Date"])
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    sample = pd.read_csv(data_dir / "sampleSubmission.csv")
    return train, test, features, stores, sample


def merge_all(df, features, stores):
    """Attach store metadata and weekly features to a train/test frame.

    `features` also carries an IsHoliday column; we drop it here and keep the
    IsHoliday that ships with train/test to avoid a duplicated column after the
    merge.
    """
    out = df.merge(stores, on="Store", how="left")
    feat = features.drop(columns=[c for c in ["IsHoliday"] if c in features.columns])
    out = out.merge(feat, on=["Store", "Date"], how="left")
    return out


def make_submission(test_df, preds, path="submission.csv"):
    """Build a Kaggle submission file with the required Id format.

    Id = "{Store}_{Dept}_{YYYY-MM-DD}", e.g. "1_1_2012-11-02".
    """
    sub = pd.DataFrame(
        {
            "Id": (
                test_df["Store"].astype(str)
                + "_"
                + test_df["Dept"].astype(str)
                + "_"
                + pd.to_datetime(test_df["Date"]).dt.strftime("%Y-%m-%d")
            ),
            "Weekly_Sales": preds,
        }
    )
    sub.to_csv(path, index=False)
    return sub