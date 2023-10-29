import re
from pipeline_analyzer.jn_analyzer.utils import debug_print
from pipeline_analyzer.jn_analyzer.constants import (
    VECTORIZERS,
    CLASSIFIERS,
    ALL_TAGS,
    ACTIVITY,
    EXPLICIT_MODELS,
    KEYWORDS
)

CELLS = None
pattern_constants = re.compile(r"^[A-Z]{1,}(?:[_A-Z0-9]{1,})=.{1,}$")
pattern_hardcoded_constant = re.compile(r"\bCONST\b")
pattern_hardcoded_setup = re.compile(r"\bSETUP\b")


# If a cell has a match for two heuristics, we can suggest to split the cell into subcells
def init_heuristics_dict(cell_count: int):
    """This function takes the number of cells and creates as many dictionaries."""
    cells = {}
    for i in range(cell_count):
        cells[i] = {
            "cell_number": i,
            "cell_type": "unknown",
            "cell_output_type": "unknown",
            "number_of_tags": "unknown",
            "activities": {
                ACTIVITY.SETUP_NOTEBOOK: -999,
                ACTIVITY.INGEST_DATA: -999,
                ACTIVITY.VALIDATE_DATA: -999,
                ACTIVITY.PROCESS_DATA: -999,
                ACTIVITY.TRAIN_MODEL: -999,
                ACTIVITY.EVALUATE_MODEL: -999,
                ACTIVITY.TRANSFER_RESULTS: -999,
                ACTIVITY.VISUALIZE_DATA: -999,
            },
        }
    global CELLS
    CELLS = cells
    debug_print("Initialized dictionaries for each notebook cell.")


def setup_notebook_heuristic_hybrid(content: str) -> int:
    if KEYWORDS.SETUP in content:
        return 1
    else:
        classifier = CLASSIFIERS[ACTIVITY.SETUP_NOTEBOOK]
        vectorizer = VECTORIZERS[ACTIVITY.SETUP_NOTEBOOK]
        vectorized_cell = vectorizer.transform([content])
        prediction = classifier.predict(vectorized_cell)
        if prediction[0] == 1:
            return 1
        else:
            return 0


def validate_data_heuristic_hybrid(content: str):
    if KEYWORDS.VALIDATION in content:
        return 1
    else:
        classifier = CLASSIFIERS[ACTIVITY.VALIDATE_DATA]
        vectorizer = VECTORIZERS[ACTIVITY.VALIDATE_DATA]
        vectorized_cell = vectorizer.transform([content])
        prediction = classifier.predict(vectorized_cell)
        if prediction == 1:
            return 1
        else:
            return 0


def visualize_data_heuristic_hybrid(content: str, cell_number: int):
    if CELLS[cell_number]["cell_output_type"] == "display_data":
        return 1
    else:
        classifier = CLASSIFIERS[ACTIVITY.VISUALIZE_DATA]
        vectorizer = VECTORIZERS[ACTIVITY.VISUALIZE_DATA]
        vectorized_cell = vectorizer.transform([content])
        prediction = classifier.predict(vectorized_cell)
        if prediction == 1:
            return 1
        else:
            return 0


def create_cell_properties(cell_number: int, cell: dict):
    CELLS[cell_number]["cell_type"] = cell["cell_type"]
    CELLS[cell_number]["cell_number"] = cell_number
    if "outputs" in cell and len(cell["outputs"]) > 0:
        CELLS[cell_number]["cell_output_type"] = cell["outputs"][0]["output_type"]
    else:
        CELLS[cell_number]["cell_output_type"] = "not_existent"


def machine_learning_prediction_hybrid(cell_number: int, text_as_list: list) -> int:
    sentence = " ".join(text_as_list)
    if len(text_as_list) == 0:
        return

    if setup_notebook_heuristic_hybrid(sentence) == 1:
        CELLS[cell_number]["activities"].update({ACTIVITY.SETUP_NOTEBOOK: float(1)})
    if visualize_data_heuristic_hybrid(sentence, cell_number) == 1:
        CELLS[cell_number]["activities"].update({ACTIVITY.VISUALIZE_DATA: float(1)})
    if validate_data_heuristic_hybrid(sentence) == 1:
        CELLS[cell_number]["activities"].update({ACTIVITY.VALIDATE_DATA: float(1)})
    sentence = " ".join(text_as_list)
    for tag in EXPLICIT_MODELS.keys():
        sentence_transformed = VECTORIZERS[tag].transform([sentence])
        prediciton = CLASSIFIERS[tag].predict(sentence_transformed)

        if float(CELLS[cell_number]["activities"][ALL_TAGS[tag]]) < prediciton[0]:
            CELLS[cell_number]["activities"].update({ALL_TAGS[tag]: prediciton[0]})


def run_source_nlp_hybrid_heuristics(cell_number: int, cell_source: list):
    machine_learning_prediction_hybrid(cell_number, cell_source)


def get_current_heuristic_dict() -> dict:
    return CELLS


def reset_heuristics_dict():
    # When assigning a global variable to a new value we need the global keyword otherwise we create a new local variable
    global CELLS
    CELLS = {}
