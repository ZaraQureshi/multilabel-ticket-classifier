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
    return parser.parse_args()


def load_new_messages(path):
    print("Path: ",path)
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

    log_file = Config.OUTPUTS_DIR / "logs" / f"predict_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),        # still prints to console
            logging.FileHandler(log_file),  # ALSO writes to disk
        ],
    )

    model_names = list(MODEL_REGISTRY.keys()) 

    

    logger.info("Loading new messages from %s", args.input)
    df = load_new_messages(args.input)

    logger.info("Applying text preprocessing")
    df_clean = clean_text_columns(df.copy())

    
   
    output_df = pd.DataFrame({
        "message_id": df["message_id"],
        "input_text": df[Config.TICKET_SUMMARY].fillna("").astype(str)
        + " | "
        + df[Config.INTERACTION_CONTENT].fillna("").astype(str),
    })
    for model_name in model_names:
        model_path = Config.ARTIFACTS_DIR / f"model_{model_name}.joblib"
        vectorizer_path = Config.ARTIFACTS_DIR / f"vectorizer_{model_name}.joblib"

        if not model_path.exists() or not vectorizer_path.exists():
            logger.error("Missing artifacts for %s: %s / %s", model_name, model_path, vectorizer_path)
            continue

        logger.info("Loading artifacts for %s", model_name)
        model = MODEL_REGISTRY[model_name].load(model_path)
        featurizer = TextFeaturizer.load(vectorizer_path)

        logger.info("Transforming text for %s", model_name)
        X = featurizer.transform(df_clean)

        logger.info("Generating predictions for %s", model_name)
        predictions = model.predict(X)
        confidences = model.predict_proba_max(X)

        for col in model.target_cols:
            output_df[f"predicted_{col}_{model_name}"] = predictions[col]
            output_df[f"confidence_{col}_{model_name}"] = confidences[col].round(4)


        output_df["prediction_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(" Wrote %d predictions to %s", len(output_df), output_path)


if __name__ == "__main__":
    main()