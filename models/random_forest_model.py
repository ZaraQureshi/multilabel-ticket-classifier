import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from Config import Config
from model.base import BaseModel


class RandomForestModel(BaseModel):
    name="random_forest"

    def __init__(self, target_cols=None, **params):
        self.target_cols=list(target_cols or Config.TYPE_COLS)
        #unpack Config.RANDOM_FOREST_PARAMS dictionary to keyword args
        rf_params={**Config.RANDOM_FOREST_PARAMS, **params}
        base_estimator = RandomForestClassifier(random_state=Config.RANDOM_SEED, **rf_params)
        self.model = MultiOutputClassifier(base_estimator)
        self.label_encoders = {col: LabelEncoder() for col in self.target_cols}

    def _encode(self, y_df, fit):
        #create empty numeric array with no. of rows as rows and no. of target columns as columns
        encoded = np.zeros((len(y_df), len(self.target_cols)), dtype=int)
        #loop through each target column with i as the index and col as the actual column name
        for i, col in enumerate(self.target_cols):
            # get the LabelEncoder for that column from the label_encoders assigned in init
            encoder = self.label_encoders[col]

            encoded[:, i] = encoder.fit_transform(y_df[col]) if fit else encoder.transform(y_df[col])
        return encoded
    
    def fit(self, X, y_df) -> None:
        y_encoded = self._encode(y_df, fit=True)
        self.model.fit(X, y_encoded)
    
    def predict(self, X) -> dict:
        y_pred_encoded = self.model.predict(X)
        predictions = {}
        for i, col in enumerate(self.target_cols):
            predictions[col] = self.label_encoders[col].inverse_transform(y_pred_encoded[:, i])
        return predictions
    
    def predict_proba_max(self, X) -> dict:
        probas = self.model.predict_proba(X)  # list of arrays, one per target column
        confidences = {}
        for i, col in enumerate(self.target_cols):
            confidences[col] = probas[i].max(axis=1)
        return confidences

    def save(self, path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "label_encoders": self.label_encoders,
                "target_cols": self.target_cols,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "RandomForestModel":
        payload = joblib.load(path)
        obj = cls(target_cols=payload["target_cols"])
        obj.model = payload["model"]
        obj.label_encoders = payload["label_encoders"]
        return obj
