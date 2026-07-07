import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd

from Config import Config
from embeddings import TextFeaturizer
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from preprocess import clean_text_columns


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "logistic_regression": LogisticRegressionModel,
}
REQUIRED_COLUMNS = [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference on new customer messages")
    parser.add_argument("--input", required=True, help="Path to a CSV of new messages")
    parser.add_argument("--output", required=True, help="Path to write the predictions CSV")
    # parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default=Config.MODEL_NAME)
    return parser.parse_args()


def load_new_messages(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    if "message_id" not in df.columns:
        logger.info("No 'message_id' column found; generating one from row index")
        df = df.reset_index().rename(columns={"index": "message_id"})

    return df

def main():
    args = parse_args()

    model_path = Config.ARTIFACTS_DIR / f"model_{Config.MODEL_NAME}.joblib"
    vectorizer_path = Config.ARTIFACTS_DIR / "vectorizer.joblib"
    if not model_path.exists() or not vectorizer_path.exists():
        logger.error(
            "Missing model or vectorizer artifact (%s / %s). Run training first: "
            "python -m src.train --model %s",
            model_path,
            vectorizer_path,
            Config.MODEL_NAME,
        )
    
    logger.info("=== Loading model and vectorizer artifacts ===")
    model = MODEL_REGISTRY[Config.MODEL_NAME].load(model_path)
    featurizer = TextFeaturizer.load(vectorizer_path)

    logger.info("=== Loading new messages from %s ===", args.input)
    df = load_new_messages(args.input)

    logger.info("=== Applying training-consistent text preprocessing ===")
    df_clean = clean_text_columns(df.copy())

    
    logger.info("=== Transforming text with the fitted (not refit) vectorizer ===")
    X = featurizer.transform(df_clean)

    logger.info("=== Generating predictions ===")
    predictions = model.predict(X)
    confidences = model.predict_proba_max(X)

    output_df = pd.DataFrame(
        {
            "message_id": df["message_id"],
            "input_text": df[Config.TICKET_SUMMARY].fillna("").astype(str)
            + " | "
            + df[Config.INTERACTION_CONTENT].fillna("").astype(str),
        }
    )
    for col in model.target_cols:
        output_df[f"predicted_{col}"] = predictions[col]
        output_df[f"confidence_{col}"] = confidences[col].round(4)


    output_df["model_name"] = args.model
    output_df["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("=== Wrote %d predictions to %s ===", len(output_df), output_path)


if __name__ == "__main__":
    main()