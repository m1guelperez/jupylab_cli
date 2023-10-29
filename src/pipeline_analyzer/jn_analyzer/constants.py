# TODO: Eventually Load froam external config file

PATH_TO_JSON_RESOURCE_INPUT_DIR = (
    "./src/pipeline_analyzer/jn_analyzer/resources/inputs/"
)
PATH_TO_JSON_RESOURCE_OUTPUT_DIR = (
    "./src/pipeline_analyzer/jn_analyzer/resources/outputs/"
)
PATH_TO_JSON_RESOURCE_BACKUP_DIR = (
    "./src/pipeline_analyzer/jn_analyzer/resources/backups/"
)
PATH_TO_NEW_TRAINED_MODELS = (
    "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/"
)
PATH_TO_TRAINING_DIR = "./src/pipeline_analyzer/jn_analyzer/resources/training_data/"
PATH_TO_VALIDATION_DIR = (
    "./src/pipeline_analyzer/jn_analyzer/resources/validation_data/"
)
PATH_TO_HEADERGEN_EVALUATION_DIR = (
    "./src/pipeline_analyzer/jn_analyzer/resources/headergen_data/"
)
PATH_TO_MODELS_DIR = "./src/pipeline_analyzer/jn_analyzer/resources/models/"

REGEX = r"(?u)[a-zA-Z]{1,}|[=[\]_]"
DEBUG_MODE = [True]
INSERT_HEADERS = [True]
VECTORIZERS = {}
CLASSIFIERS = {}


class ACTIVITY:
    SETUP_NOTEBOOK = "setup_notebook"
    INGEST_DATA = "ingest_data"
    VALIDATE_DATA = "validate_data"
    PROCESS_DATA = "process_data"
    TRAIN_MODEL = "train_model"
    EVALUATE_MODEL = "evaluate_model"
    TRANSFER_RESULTS = "transfer_results"
    VISUALIZE_DATA = "visualize_data"
    CHECK_RESULTS = "check_results"


ALL_TAGS = {
    ACTIVITY.SETUP_NOTEBOOK: "setup_notebook",
    ACTIVITY.INGEST_DATA: "ingest_data",
    ACTIVITY.VALIDATE_DATA: "validate_data",
    ACTIVITY.PROCESS_DATA: "process_data",
    ACTIVITY.TRAIN_MODEL: "train_model",
    ACTIVITY.EVALUATE_MODEL: "evaluate_model",
    ACTIVITY.TRANSFER_RESULTS: "transfer_results",
    ACTIVITY.VISUALIZE_DATA: "visualize_data",
    ACTIVITY.CHECK_RESULTS: "check_results",
}

EXPLICIT_MODELS = {
    ACTIVITY.INGEST_DATA: "ingest_data",
    ACTIVITY.VALIDATE_DATA: "validate_data",
    ACTIVITY.PROCESS_DATA: "process_data",
    ACTIVITY.TRAIN_MODEL: "train_model",
    ACTIVITY.EVALUATE_MODEL: "evaluate_model",
    ACTIVITY.TRANSFER_RESULTS: "transfer_results",
}
