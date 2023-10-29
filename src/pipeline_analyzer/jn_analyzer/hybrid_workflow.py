import json
import copy
import os
import timeit
import shutil
from tqdm import tqdm
from pipeline_analyzer.jn_analyzer.constants import (
    PATH_TO_JSON_RESOURCE_INPUT_DIR,
    PATH_TO_JSON_RESOURCE_OUTPUT_DIR,
    PATH_TO_JSON_RESOURCE_BACKUP_DIR,
    DEBUG_MODE,
    INSERT_HEADERS,
)
from pipeline_analyzer.jn_analyzer.heuristics import (
    get_current_heuristic_dict,
    init_heuristics_dict,
    reset_heuristics_dict,
    create_cell_properties,
    run_source_nlp_hybrid_heuristics,
)
from pipeline_analyzer.jn_analyzer.utils import (
    make_labels,
    path_trim_extension,
    preprocess_source_cell_nlp,
    debug_print,
    delete_kaggle_cells,
    load_pre_trained_models,
)


def pre_process_notebook_nlp(notebook_name: str, notebook: dict):
    if DEBUG_MODE[0]:
        print("Saving backup of notebook...")
        with open(PATH_TO_JSON_RESOURCE_BACKUP_DIR + notebook_name + "json", "w") as f:
            json.dump(notebook, f, indent=4, ensure_ascii=False)
        print("Finished saving backup of notebook.")
    debug_print("Starting to pre-process notebook.\nInitializing heuristics...")
    init_heuristics_dict(len(notebook["cells"]))
    debug_print("Finished initializing heuristics.\nCleaning notebook from newlines...")
    for cell_number, cell in enumerate(notebook["cells"]):
        create_cell_properties(cell_number, cell)
        cell["cell_number"] = cell_number
        if cell["cell_type"] == "code":
            if len(cell["source"]) > 0:
                cell["source_old"] = cell["source"]
                cell["source"] = preprocess_source_cell_nlp(cell["source"])
    if DEBUG_MODE[0]:
        with open(
            PATH_TO_JSON_RESOURCE_OUTPUT_DIR
            + notebook_name
            + "_nlp_hybrid_preprocessed.json",
            "w",
        ) as myfile:
            json.dump(notebook, myfile, indent=4)
        print("Finished cleaning notebook from newlines.")


# Creates a dict with heuristics for each cell and returns the result after we replied the heuristics on the notebook
def analyze_notebook_nlp_hybrid(preprocessed_notebook: dict):
    """Applies the nlp hybrid based solution and will only consider the source code of a cell."""
    for cell_number, cell in enumerate(preprocessed_notebook["cells"]):
        if cell["cell_type"] == "code":
            if len(cell["source"]) > 0:
                run_source_nlp_hybrid_heuristics(cell_number, cell["source"])
    debug_print("Finished applying all heuristics on each cell.")


def post_process_notebook_nlp(file_name: str, original_notebook: dict):
    """Creates the labeling of the cells."""
    debug_print("Starting post-processing...\nStarting to create labels...")
    heuristics_dict = get_current_heuristic_dict()
    make_labels(heuristics_dict, original_notebook)
    # show labels by default, could be made optional
    original_notebook["metadata"]["celltoolbar"] = "Tags"
    if DEBUG_MODE[0]:
        path_to_json = (
            PATH_TO_JSON_RESOURCE_OUTPUT_DIR + file_name + "_nlp_labeled.json"
        )
        debug_print("Finished creating labels.\nSaving notebook to disk...")
        with open(path_to_json, "w") as f:
            json.dump(original_notebook, f, indent=4, ensure_ascii=False)
    path_to_ipynb = PATH_TO_JSON_RESOURCE_OUTPUT_DIR + file_name + "_nlp_labeled.ipynb"
    with open(path_to_ipynb, "w") as f:
        json.dump(original_notebook, f, indent=4, ensure_ascii=False)
    debug_print("Finished saving notebook to disk.")


def start_pipeline_hybrid(basepath: str, debug_mode: bool, headers: bool):
    """Starts the pipeline for nlp based solution."""
    DEBUG_MODE[0] = debug_mode
    INSERT_HEADERS[0] = headers
    with tqdm(total=len(os.listdir(basepath)), colour="blue") as pbar:
        for entry in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath, entry)):
                path = os.path.join(basepath, entry)
                try:
                    with open(path, "r") as f:
                        notebook = json.load(f)
                    original_notebook = copy.deepcopy(notebook)
                except json.decoder.JSONDecodeError:
                    debug_print("Error: Could not load notebook: " + path)
                    continue
                debug_print("Starting pipeline for notebook: " + path)
                # Python passes dicts by reference therefore we do not need to save the result to a new variable
                debug_print(path_trim_extension(path).split("/")[-1])
                pre_process_notebook_nlp(
                    path_trim_extension(path).split("/")[-1], notebook
                )
                analyze_notebook_nlp_hybrid(notebook)
                post_process_notebook_nlp(
                    path_trim_extension(path).split("/")[-1], original_notebook
                )
                reset_heuristics_dict()
            pbar.update(1)


def ml_data_hybrid(pre_processed_notebook: dict, ML_CONTENT: dict):
    heuristics_dict = get_current_heuristic_dict()
    make_labels(heuristics_dict, pre_processed_notebook)
    for cell in pre_processed_notebook["cells"]:
        source_content = {"tags": [], "content": [], "output_type": ""}
        if cell["cell_type"] == "code":
            if len(cell["source"]) > 0 and "tags" in cell["metadata"]:
                source_content["tags"] = cell["metadata"]["tags"]
                source_content["content"] = cell["source"]
                source_content["content_old"] = (
                    lambda x: x if isinstance(x, list) else x.split("\n")
                )(cell["source_old"])
                if len(cell["outputs"]) > 0:
                    source_content["output_type"] = cell["outputs"][0]["output_type"]
                else:
                    source_content["output_type"] = "not_existent"
                ML_CONTENT["source"].append(source_content)


def benchmark_hybrid(debug_mode: bool, headers: bool):
    assert len(os.listdir(PATH_TO_JSON_RESOURCE_INPUT_DIR)) == 1000
    num_executions = 10
    total_time = 0
    DEBUG_MODE[0] = debug_mode
    INSERT_HEADERS[0] = headers
    load_pre_trained_models()
    with tqdm(total=num_executions, colour="red") as pbar:
        for _ in range(num_executions):
            start_time = timeit.default_timer()
            start_pipeline_hybrid(PATH_TO_JSON_RESOURCE_INPUT_DIR, debug_mode, headers)
            end_time = timeit.default_timer()
            execution_time = end_time - start_time
            total_time += execution_time
            shutil.rmtree(PATH_TO_JSON_RESOURCE_OUTPUT_DIR)
            os.mkdir(PATH_TO_JSON_RESOURCE_OUTPUT_DIR)
            shutil.rmtree(PATH_TO_JSON_RESOURCE_BACKUP_DIR)
            os.mkdir(PATH_TO_JSON_RESOURCE_BACKUP_DIR)
            pbar.update(1)

    average_time = total_time / num_executions
    print("Average execution time of the function is: ", average_time)


def start_pipeline_hybrid_cli_test(
    basepath: str, debug_mode: bool, headers: bool, ML_CONTENT: dict
):
    """Starts the pipeline for nlp based solution."""
    DEBUG_MODE[0] = debug_mode
    INSERT_HEADERS[0] = headers
    with tqdm(total=len(os.listdir(basepath)), colour="blue") as pbar:
        for entry in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath, entry)):
                path = os.path.join(basepath, entry)
                try:
                    with open(path, "r") as f:
                        notebook = json.load(f)
                    original_notebook = copy.deepcopy(notebook)
                except json.decoder.JSONDecodeError:
                    debug_print("Error: Could not load notebook: " + path)
                    continue
                debug_print("Starting pipeline for notebook: " + path)
                # Python passes dicts by reference therefore we do not need to save the result to a new variable
                debug_print(path_trim_extension(path).split("/")[-1])
                pre_process_notebook_nlp(
                    path_trim_extension(path).split("/")[-1], notebook
                )
                analyze_notebook_nlp_hybrid(notebook)
                post_process_notebook_nlp(
                    path_trim_extension(path).split("/")[-1], original_notebook
                )
                ml_data_hybrid(notebook, ML_CONTENT)
                reset_heuristics_dict()
            pbar.update(1)


def create_cli_inference_file(
    path_to_notebooks: str = PATH_TO_JSON_RESOURCE_INPUT_DIR,
    output_file: str = "cli_run.json",
    debug_mode: bool = False,
    headers: bool = False,
):
    ML_CONTENT = {"source": []}
    DEBUG_MODE[0] = debug_mode
    INSERT_HEADERS[0] = headers
    load_pre_trained_models()
    start_pipeline_hybrid_cli_test(path_to_notebooks, debug_mode, headers, ML_CONTENT)
    ML_CONTENT = delete_kaggle_cells(ML_CONTENT)
    with open(output_file, "w") as f:
        json.dump(ML_CONTENT, f, indent=4, ensure_ascii=False)
