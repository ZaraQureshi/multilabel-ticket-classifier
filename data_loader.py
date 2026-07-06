import logging

import pandas as pd

from Config import Config 

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = [
    Config.TICKET_SUMMARY,
    Config.INTERACTION_CONTENT,
    "Type 1",
    "Type 2",
    "Type 3",
    "Type 4",
]

def validate_schema(df, source_name):
    missing=[c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{source_name}: file loaded with zero rows")

    null_frac = df[Config.INTERACTION_CONTENT].isna().sum()
    if null_frac!=0:
        logger.warning(f" {Config.INTERACTION_CONTENT} has {null_frac} null values")

def load_raw_data():
    frames = []

    for name,path in Config.DATA_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")

        df = pd.read_csv(path)
        validate_schema(df, name)

        df = df.rename(
            columns={"Type 1": "y1", "Type 2": "y2", "Type 3": "y3", "Type 4": "y4"}
        )
        df["source_file"] = name
        #loads the files and concatenates them
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d source file(s)", len(df), len(frames))
    return df
