from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X, y_df) -> None:
        """Fit the model on features X and a DataFrame of target columns."""

    @abstractmethod
    def predict(self, X) -> dict:
        """Return {target_col: predicted_labels_array} for each target column."""

    @abstractmethod
    def predict_proba_max(self, X) -> dict:
        """Return {target_col: confidence_array} — the top predicted class's
        probability for each row, per target column."""

    @abstractmethod
    def save(self, path) -> None:
        """Persist the fitted model (and any label encoders) to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path) -> "BaseModel":
        """Load a previously saved model from disk."""
