import json
import logging

import pandas as pd
from sklearn.metrics import classification_report


logger = logging.getLogger(__name__)


def evaluate_multi_output(y_true_df: pd.DataFrame, y_pred: dict, target_cols) -> dict:
    report = {}
    for col in target_cols:
        y_true = y_true_df[col].values
        y_hat = y_pred[col]
        report[col] = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
        macro_f1 = report[col]["macro avg"]["f1-score"]
        acc = report[col].get("accuracy")
        logger.info("Target '%s': accuracy=%.3f macro-F1=%.3f", col, acc, macro_f1)
    return report

def save_metrics_json(report: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics JSON to %s", path)

def report_to_dataframe(report: dict) -> pd.DataFrame:
    rows = []
    for target_col, target_report in report.items():
        for label, metrics in target_report.items():
            if isinstance(metrics, dict):  # skip the scalar "accuracy" key
                rows.append({"target": target_col, "label": label, **metrics})
    return pd.DataFrame(rows)

def save_classification_report_csv(report: dict, path) -> None:
    df = report_to_dataframe(report)
    df.to_csv(path, index=False)
    logger.info("Saved classification report CSV to %s", path)
