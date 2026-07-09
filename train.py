import argparse
import logging

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-label customer interaction classifier")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default=Config.MODEL_NAME)
    parser.add_argument("--test-size", type=float, default=Config.TEST_SIZE)
    return parser.parse_args()

def main():
    args=parse_args()
    Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Load raw data")
    df = load_raw_data()

    logger.info("Clean text")
    df = clean_text_columns(df)

    logger.info("Filtering rare classes (min %d samples per class)", Config.MIN_CLASS_SAMPLES)
    df = filter_rare_classes(df, Config.TYPE_COLS, Config.MIN_CLASS_SAMPLES)

    if len(df) < 20:
        logger.error("Not enough data remaining after filtering (%d rows)", len(df))
        
    logger.info("Splitting train/test (test_size=%.2f)", Config.TEST_SIZE)

    #data split into train and test and after that tf-idf will be applied 
    train_df, test_df = train_test_split(df, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED)
    logger.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    logger.info("Fitting TF-IDF vectorizer on training data only")
    featurizer = TextFeaturizer()
    X_train = featurizer.fit_transform(train_df)
    X_test = featurizer.transform(test_df)  # transform only - no leakage

    logger.info("Training model: %s ", args.model)
    model = MODEL_REGISTRY[args.model](target_cols=Config.TYPE_COLS)
    model.fit(X_train, train_df[Config.TYPE_COLS])

    logger.info("Evaluating on test set")
    y_pred = model.predict(X_test)
    report = evaluate_multi_output(test_df, y_pred, Config.TYPE_COLS)

    metrics_path = Config.OUTPUTS_DIR /f"metrics_{args.model}.json"
    report_csv_path = Config.OUTPUTS_DIR /f"classification_report_{args.model}.csv"
    save_metrics_json(report, metrics_path)
    save_classification_report_csv(report, report_csv_path)

    model_path = Config.ARTIFACTS_DIR / f"model_{args.model}.joblib"
    vectorizer_path = Config.ARTIFACTS_DIR / "vectorizer.joblib"
    model.save(model_path)
    featurizer.save(vectorizer_path)

    logger.info("Model saved to: %s", model_path)
    logger.info("Vectorizer saved to: %s", vectorizer_path)
    logger.info("Metrics saved to: %s", metrics_path)
    logger.info("Report CSV saved to: %s", report_csv_path)


if __name__ == "__main__":
    main()
