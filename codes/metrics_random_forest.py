import gzip
import pickle

import pandas as pd
from remissia.utils import attributes, get_path

path = get_path()

feat_imp = []
metrics_values = []
metrics_names = ["accuracy", "precision", "recall", "f2", "kappa", "roc_auc"]
for i in range(30):

    with gzip.open(path["files"] / f"run_{i}.gz.pkl", "rb") as file:
        metrics = pickle.load(file)

    feat_imp.append(metrics["feat_imp"])
    metrics_values.append(metrics["metric"])

df = pd.DataFrame(feat_imp, columns=attributes)
metrics_values = pd.DataFrame(metrics_values, columns=metrics_names)

df_to_export = pd.concat((df.mean(axis=0), df.std(axis=0)), axis=1)
df_to_export = df_to_export.sort_values(0, ascending=False)
df_to_export.to_csv(path["files"] / "attributes.csv")
metrics_to_export = pd.concat(
    (metrics_values.mean(axis=0), metrics_values.std(axis=0)), axis=1
)
metrics_to_export.to_csv(path["files"] / "metrics.csv")
