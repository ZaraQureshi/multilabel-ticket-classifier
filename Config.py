from pathlib import Path


class Config:
    """Basic configuration for the starter prototype.

    This file is intentionally simple. Students may improve configuration
    management as part of the assessment.
    """

    #Paths
    BASE_DIR = Path(__file__).resolve().parent
    print(BASE_DIR)
    DATA_DIR = BASE_DIR / "data"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    OUTPUTS_DIR = BASE_DIR / "outputs"

    DATA_FILES = {
        "AppGallery": DATA_DIR / "AppGallery.csv",
        "Purchasing": DATA_DIR / "Purchasing.csv",
        
    }


    # Input text columns
    TICKET_SUMMARY = "Ticket Summary"
    INTERACTION_CONTENT = "Interaction content"

    # Value substituted for missing labels (e.g. y3/y4 not always present).
    # Treated as its own valid class rather than dropping the row.
    MISSING_LABEL_FILL = "Not Applicable"

    # Label columns after renaming original Type 1-Type 4 columns
    TYPE_COLS = ["y2", "y3", "y4"]

    # Core assessment target label.
    # y3 and y4 are available in the dataset but are not required for the core task.
    CLASS_COL = "y2"

    # Used by the existing prototype to run separate experiments per Type 1 group.
    GROUPED = "y1"

    # Data filtering / splitting
    MIN_CLASS_SAMPLES = 3
    TEST_SIZE = 0.2
    RANDOM_SEED = 0

    # Feature extraction
    TFIDF_MAX_FEATURES = 2000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95

    # Modelling
    MODEL_NAME = "random_forest"
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 300,
        "class_weight": "balanced_subsample",
    }

    LOGISTIC_REGRESSION_PARAMS = {
        "max_iter": 2000,
        "class_weight": "balanced",
    }