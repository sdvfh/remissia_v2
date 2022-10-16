import gzip
import pickle

import pandas as pd
from remissia.utils import get_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

path = get_path()
target = "NO_REMISSION"

file_parquet = list(path["dataset"].rglob("*.parquet"))[0]
df = pd.read_parquet(file_parquet)

df = df.drop(columns=["DEAD_DUE_CANCER", "TIME_SURVIVAL"])

attributes = df.drop(columns=[target]).columns.tolist()

list_to_save = []
for i in range(30):
    print(i)
    train, test = train_test_split(
        df, test_size=0.25, random_state=i, stratify=df[target]
    )
    train = train.drop_duplicates(subset=attributes)

    train_hash = pd.util.hash_pandas_object(train[attributes], index=False)
    test_hash = pd.util.hash_pandas_object(test[attributes], index=False)

    train = train[~train_hash.isin(test_hash)]
    ml = RandomForestClassifier(random_state=i, verbose=1, n_jobs=-1)

    ml.fit(train.drop(columns=[target]), train[target])

    y_pred = ml.predict_proba(test.drop(columns=[target]))[:, 1]
    y_true = test[target]
    y_pred_round = y_pred.copy()
    y_pred_round = y_pred_round.round()
    metric = {
        "accuracy": balanced_accuracy_score(y_true, y_pred_round),
        "precision": precision_score(y_true, y_pred_round),
        "recall": recall_score(y_true, y_pred_round),
        "f2": fbeta_score(y_true, y_pred_round, beta=2),
        "kappa": cohen_kappa_score(y_true, y_pred_round),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }

    to_save = {
        "metric": metric.copy(),
        "feat_imp": ml.feature_importances_,
        "y_pred": y_pred.copy(),
        "y_true": test[target],
    }

    with gzip.open(path["files"] / f"run_{i}.gz.pkl", "wb") as file:
        pickle.dump(to_save, file)
