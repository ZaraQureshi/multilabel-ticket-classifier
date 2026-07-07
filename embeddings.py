import logging
import random

import joblib
import numpy as np
import pandas as pd
from Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer

seed = 0
random.seed(seed)
np.random.seed(seed)
logger = logging.getLogger(__name__)

class TextFeaturizer:
    def __init__(self, max_features=None,min_df=None,max_df=None):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features or Config.TFIDF_MAX_FEATURES,
            min_df=min_df or Config.TFIDF_MIN_DF,
            max_df=max_df or Config.TFIDF_MAX_DF,
        )
        self._fitted = False

    @staticmethod
    def _combine_text(df):
        return (
            df[Config.TICKET_SUMMARY].fillna("").astype(str)
            + " "
            + df[Config.INTERACTION_CONTENT].fillna("").astype(str)
        )

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        text = self._combine_text(df)
        X = self.vectorizer.fit_transform(text).toarray()
        self._fitted = True
        logger.info("Fitted TF-IDF vectorizer: vocabulary size = %d", len(self.vectorizer.vocabulary_))
        return X
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "TextFeaturizer must be fit (via fit_transform) before calling transform. "
                "This usually means you're trying to run inference without a trained/saved vectorizer."
            )
        text = self._combine_text(df)
        return self.vectorizer.transform(text).toarray()
    def save(self, path) -> None:
        joblib.dump(self.vectorizer, path)
        logger.info("Saved vectorizer to %s", path)

    @classmethod
    def load(cls, path) -> "TextFeaturizer":
        obj = cls()
        obj.vectorizer = joblib.load(path)
        obj._fitted = True
        return obj


