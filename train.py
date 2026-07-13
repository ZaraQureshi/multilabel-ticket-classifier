import datetime
import logging
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from Config import Config
from data_loader import load_raw_data
from embeddings import TextFeaturizer
from evaluation import evaluate_multi_output, save_classification_report_csv, save_metrics_json
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from preprocess import clean_text_columns, filter_rare_classes


logger = logging.getLogger(__name__)
MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "logistic_regression": LogisticRegressionModel,
}


import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrices(y_true_df: pd.DataFrame, y_pred: dict, target_cols, output_dir) -> None:
    
    for col in target_cols:
        y_true = y_true_df[col].values
        y_hat = y_pred[col]
        labels = sorted(set(y_true) | set(y_hat))

        cm = confusion_matrix(y_true, y_hat, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        csv_path = output_dir / f"confusion_matrix_{col}.csv"
        cm_df.to_csv(csv_path)
        logger.info("Saved confusion matrix CSV to %s", csv_path)

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {col}")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        png_path = output_dir / f"confusion_matrix_{col}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix plot to %s", png_path)


def main():
    Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = Config.OUTPUTS_DIR / "logs" / f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),           # still prints to console
        logging.FileHandler(log_file),     # ALSO writes to disk
        ],
    )
    logger.info("Load raw data")
    df = load_raw_data()

    logger.info("Clean text")
    df = clean_text_columns(df)

    logger.info("Filtering rare classes (min %d samples are required per class)", Config.MIN_CLASS_SAMPLES)
    df = filter_rare_classes(df, Config.TYPE_COLS, Config.MIN_CLASS_SAMPLES)

    if len(df) < 20:
        logger.error("Not enough data remaining after filtering (%d rows)", len(df))
        
    logger.info("Splitting train/test (test_size=%.2f)", Config.TEST_SIZE)

    #data split into train and test and after that tf-idf will be applied 
    
    for col in Config.TYPE_COLS:
        counts = df[col].value_counts().sort_index()
        percs = (counts / len(df) * 100).round(1)
        table = pd.DataFrame({
            f"train_count": counts,
            f"train_%": percs,
        })
        print("Before:", table)
    y_indicator = pd.get_dummies(df[Config.TYPE_COLS])

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
    )
    train_idx, test_idx = next(splitter.split(df, y_indicator.to_numpy()))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print("\n=== Stratification check: class proportions (%) per target column ===")
    for col in Config.TYPE_COLS:
        comparison = pd.DataFrame({
            "train_%": (train_df[col].value_counts(normalize=True) * 100).round(1),
            "test_%": (test_df[col].value_counts(normalize=True) * 100).round(1),
        }).sort_index()
        print(f"\n--- {col} ---")
        print(comparison)
    # train_df, test_df = train_test_split(df, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED)
    logger.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    logger.info("Fitting TF-IDF vectorizer on training data only")
    featurizer = TextFeaturizer()
    X_train = featurizer.fit_transform(train_df)
    X_test = featurizer.transform(test_df)  # transform only - no leakage

    y_test = test_df[Config.TYPE_COLS]
    for model_name, model_cls in MODEL_REGISTRY.items():
        logger.info("Training model: %s", model_name)
        model = model_cls(target_cols=Config.TYPE_COLS)
        model.fit(X_train, train_df[Config.TYPE_COLS])

        logger.info("Evaluating %s on test set", model_name)
        y_pred = model.predict(X_test)
        # y_pred = pd.DataFrame(y_pred, columns=Config.TYPE_COLS).to_numpy()
        # print("y_true shape: %s, y_pred shape: %s", y_test.shape, y_pred.shape)


        report = evaluate_multi_output(y_test, y_pred, Config.TYPE_COLS)

        metrics_path = Config.OUTPUTS_DIR / f"metrics_{model_name}.json"
        report_csv_path = Config.OUTPUTS_DIR / f"classification_report_{model_name}.csv"
        save_metrics_json(report, metrics_path)
        save_classification_report_csv(report, report_csv_path)
        save_confusion_matrices(y_test, y_pred, Config.TYPE_COLS, Config.OUTPUTS_DIR)

        model_path = Config.ARTIFACTS_DIR / f"model_{model_name}.joblib"
        vectorizer_path = Config.ARTIFACTS_DIR / f"vectorizer_{model_name}.joblib"
        model.save(model_path)
        featurizer.save(vectorizer_path)

        logger.info("Model saved to: %s", model_path)
        logger.info("Vectorizer saved to: %s", vectorizer_path)
        logger.info("Metrics saved to: %s", metrics_path)
        logger.info("Report CSV saved to: %s", report_csv_path)



if __name__ == "__main__":
    main()
