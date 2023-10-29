import json
import os
import numpy as np
from pipeline_analyzer.jn_analyzer.utils import (
    preprocess_source_cell_nlp,
    load_pre_trained_models,
)
from pipeline_analyzer.jn_analyzer.constants import (
    CLASSIFIERS,
    VECTORIZERS,
    REGEX,
    PATH_TO_HEADERGEN_EVALUATION_DIR,
    PATH_TO_TRAINING_DIR,
    PATH_TO_MODELS_DIR,
    PATH_TO_VALIDATION_DIR,
    ACTIVITY,
    ALL_TAGS,
    EXPLICIT_MODELS,
)
import warnings

warnings.filterwarnings("ignore")
import xgboost as xgb
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report


def checkpoint_activity_heuristic_hybrid(content: str):
    if "CHECKPOINT" in content:
        return 1
    else:
        classifier = CLASSIFIERS[ACTIVITY.CHECK_RESULTS]
        vectorizer = VECTORIZERS[ACTIVITY.CHECK_RESULTS]
        vectorized_cell = vectorizer.transform([content])
        prediction = classifier.predict(vectorized_cell)
        if prediction == 1:
            return 1
        else:
            return 0


def data_visualization_heuristic_hybrid(content: str, output_type: str):
    if output_type == "display_data":
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


def setup_phase_heuristic_hybrid(content: str) -> int:
    if "SETUP" in content:
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


def print_positive_report(list_values, name):
    print(
        "Positive " + str(name) + " score: " + str(sum(list_values) / len(list_values))
    )


def print_total_average_macro_report(list_values, name):
    print(
        "Average macro "
        + str(name)
        + " score: "
        + str(sum(list_values) / len(list_values))
    )


def print_weighted_report_per_phase(report, name):
    print(
        "Average weighted "
        + name
        + " score per phase: "
        + str(report["weighted avg"][name.lower()])
    )


def print_macro_report_per_phase(report, name):
    print(
        "Average macro "
        + name
        + " score per phase: "
        + str(report["macro avg"][name.lower()])
    )


def positive_report_per_phase(report, name):
    print("Positive " + name + " score per phase: " + str(report["1"][name.lower()]))


def print_total_positive_report(list_values, name):
    print(
        "Positive " + str(name) + " score: " + str(sum(list_values) / len(list_values))
    )


def new_preprocessing_training():
    print("Creating new jsons from 120 training notebooks with new pre-processing...")
    for i in range(1, 3):
        with open(
            PATH_TO_TRAINING_DIR
            + "/training/json_processed_part1_b"
            + str(i)
            + ".json",
            "r",
            encoding="utf-8",
        ) as f:
            file_as_json = json.load(f)

        for cell in file_as_json["content"]:
            cell["source_orig"] = cell["source"]
            cell["source"] = preprocess_source_cell_nlp(cell["source"])

        with open(
            PATH_TO_TRAINING_DIR
            + "/training/"
            + "json_processed_part1_with_pre_processing_"
            + str(i)
            + ".json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(file_as_json, f, ensure_ascii=False, indent=4)

    print("Creating new jsons from validation notebooks with new pre-processing...")
    with open(
        PATH_TO_VALIDATION_DIR + "/ML_DATA_HYBRID_FINAL_COMPLETE.json",
        "r",
        encoding="utf-8",
    ) as f:
        file_as_json = json.load(f)

        for cell in file_as_json["source"]:
            cell["content_old"] = preprocess_source_cell_nlp(cell["content_old"])
    with open(
        PATH_TO_VALIDATION_DIR
        + str(datetime.now().strftime("%Y-%m-%d"))
        + "_"
        + "ML_DATA_HYBRID_FINAL_COMPLETE_NEW_PROCESSED.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(file_as_json, f, ensure_ascii=False, indent=4)


def update_pre_processing_in_ML():
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_COMPLETE.json",
        "r",
    ) as f:
        json_long = json.load(f)
    for cell in json_long["source"]:
        cell["content"] = preprocess_source_cell_nlp(cell["content_old"])
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_COMPLETE.json",
        "w",
    ) as f:
        json.dump(json_long, f, indent=4, ensure_ascii=False)

    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL.json",
        "r",
    ) as f:
        json_long = json.load(f)
    for cell in json_long["source"]:
        cell["content"] = preprocess_source_cell_nlp(cell["content_old"])
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL.json",
        "w",
    ) as f:
        json.dump(json_long, f, indent=4, ensure_ascii=False)


def create_merged():
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_SHORT.json",
        "r",
    ) as f:
        json_short = json.load(f)

    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL.json",
        "r",
    ) as f:
        json_long = json.load(f)
    print(len(json_long["source"]))
    print(len(json_short["source"]))
    assert len(json_long["source"]) == len(json_short["source"]), (
        "Length of: "
        + str(len(json_long["source"]))
        + " and "
        + str(len(json_short["source"]))
        + " not equal."
    )

    for i in range(len(json_long["source"])):
        assert (
            json_long["source"][i]["content_old"] == json_short["source"][i]["content"]
        )
        json_long["source"][i]["tags"] = json_short["source"][i]["tags"]
        json_long["source"][i]["output_type"] = json_short["source"][i]["output_type"]

    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_COMPLETE.json",
        "w",
    ) as f:
        json.dump(json_long, f, indent=4, ensure_ascii=False)


def update_final():
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_SHORT.json",
        "r",
    ) as f:
        json_small = json.load(f)
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL.json",
        "r",
    ) as f:
        json_big = json.load(f)

    for i in range(len(json_small["source"])):
        assert (
            json_small["source"][i]["content"] == json_big["source"][i]["content_old"]
        )
        json_big["source"][i]["tags"] = json_small["source"][i]["tags"]
        json_big["source"][i]["output_type"] = json_small["source"][i]["output_type"]
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL.json",
        "w",
    ) as f:
        json.dump(json_big, f, indent=4, ensure_ascii=False)


def create_new_trainings_csvs():
    """Uses the 120 training notebooks."""
    random_seed = 42
    np.random.seed(random_seed)
    print("Creating new training data..")
    with open(
        PATH_TO_TRAINING_DIR
        + "training/json_processed_part1_with_pre_processing_"
        + str(1)
        + ".json",
        "r",
        encoding="utf-8",
    ) as f:
        file_as_json = json.load(f)

    with open(
        PATH_TO_TRAINING_DIR
        + "training/json_processed_part1_with_pre_processing_"
        + str(2)
        + ".json",
        "r",
        encoding="utf-8",
    ) as f2:
        file_as_json2 = json.load(f2)

    for tag in ALL_TAGS.keys():
        df = pd.DataFrame(columns=["content", "tag", "content_original"])
        index = 0
        for cell in file_as_json["content"]:
            if tag in cell["tags"]:
                df.loc[index] = (" ".join(cell["source"]), 1, cell["source_orig"])
            else:
                df.loc[index] = (" ".join(cell["source"]), 0, cell["source_orig"])
            index += 1
        for cell in file_as_json2["content"]:
            if tag in cell["tags"]:
                df.loc[index] = (" ".join(cell["source"]), 1, cell["source_orig"])
            else:
                df.loc[index] = (" ".join(cell["source"]), 0, cell["source_orig"])
            index += 1
        # remove nan rows
        print("TRAINING CSV")
        print("Before dropping nan rows: ", df.shape)
        df = df.dropna()
        print("After dropping nan rows: ", df.shape)
        # drop empty strings
        print("Before dropping empty strings: ", df.shape)
        df = df[df["content"] != ""]
        print("After dropping empty strings: ", df.shape)
        # shuffle csv
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(
            PATH_TO_TRAINING_DIR
            + "training/csvs/"
            + tag.replace("-", "_").replace(" ", "_")
            + "_shuffled_b1_b2_concat"
            + "-"
            + str(datetime.now().strftime("%Y-%m-%d"))
            + ".csv",
            index=False,
        )


def create_final_validation_csv():
    np.random.seed(42)
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_COMPLETE.json",
        "r",
    ) as f:
        json_final = json.load(f)

    for tag in ALL_TAGS.keys():
        print("Starting with " + tag + " ...")
        df = pd.DataFrame(columns=["content", "tag", "output_type", "original_content"])
        for cell in json_final["source"]:
            if tag in cell["tags"]:
                df.loc[len(df)] = [
                    " ".join(preprocess_source_cell_nlp(cell["content_old"])),
                    1,
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                ]
            else:
                df.loc[len(df)] = [
                    " ".join(preprocess_source_cell_nlp(cell["content_old"])),
                    0,
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                ]
        print("Length before dropping dups: " + str(len(df)))
        df = df.drop_duplicates(subset=["original_content"], keep="first")
        print("Length after dropping dups: " + str(len(df)))
        df.to_csv(
            PATH_TO_VALIDATION_DIR
            + "/csvs/"
            + tag.replace("-", "_").replace(" ", "_")
            + "_final.csv",
            index=False,
        )


def create_percentage_distribution():
    ######################################
    number_of_code_cells_global = 0
    number_of_none_cells_global = 0
    number_of_checkpoints_cells_global = 0
    number_of_visualization_cells_global = 0
    number_of_evaluation_cells_global = 0
    number_of_training_cells_global = 0
    number_of_preprocessing_cells_global = 0
    number_of_validation_cells_global = 0
    number_of_ingestion_cells_global = 0
    number_of_setup_cells_global = 0
    number_of_post_development_cells_global = 0
    ######################################
    percentage_of_none_cells_per_notebook = []
    percentage_of_setup_cells_per_notebook = []
    percentage_of_ingestion_cells_per_notebook = []
    percentage_of_validation_cells_per_notebook = []
    percentage_of_preprocessing_cells_per_notebook = []
    percentage_of_training_cells_per_notebook = []
    percentage_of_evaluation_cells_per_notebook = []
    percentage_of_visualization_cells_per_notebook = []
    percentage_of_checkpoints_cells_per_notebook = []
    percentage_of_post_development_cells_per_notebook = []

    with open(
        "./analysis_per_notebook_with_sot_tags.json",
        "r",
        encoding="utf-8",
    ) as f:
        notebooks = json.load(f)
    number_of_code_cells = 0
    for notebook in notebooks["notebooks"]:
        number_of_code_cells = 0
        number_of_none_cells = 0
        number_of_setup_cells = 0
        number_of_ingestion_cells = 0
        number_of_validation_cells = 0
        number_of_preprocessing_cells = 0
        number_of_training_cells = 0
        number_of_evaluation_cells = 0
        number_of_visualization_cells = 0
        number_of_checkpoints_cells = 0
        number_of_post_development_cells = 0
        for cell in notebook["source"]:
            number_of_code_cells += 1
            number_of_code_cells_global += 1
            if "Setup Activity" in cell["tags"]:
                number_of_setup_cells += 1
                number_of_setup_cells_global += 1
            if "Data Ingestion Activity" in cell["tags"]:
                number_of_ingestion_cells += 1
                number_of_ingestion_cells_global += 1
            if "Data Validation Activity" in cell["tags"]:
                number_of_validation_cells += 1
                number_of_validation_cells_global += 1
            if "Data Pre-Processing Activity" in cell["tags"]:
                number_of_preprocessing_cells += 1
                number_of_preprocessing_cells_global += 1
            if "Model Training Activity" in cell["tags"]:
                number_of_training_cells += 1
                number_of_training_cells_global += 1
            if "Model Evaluation Activity" in cell["tags"]:
                number_of_evaluation_cells += 1
                number_of_evaluation_cells_global += 1
            if "Data Visualization Phase" in cell["tags"]:
                number_of_visualization_cells += 1
                number_of_visualization_cells_global += 1
            if "Checkpoint Activity" in cell["tags"]:
                number_of_checkpoints_cells += 1
                number_of_checkpoints_cells_global += 1
            if "Post Development Phase" in cell["tags"]:
                number_of_post_development_cells += 1
                number_of_post_development_cells_global += 1
            if "None" in cell["tags"] or len(cell["tags"]) == 0:
                number_of_none_cells += 1
                number_of_none_cells_global += 1
        percentage_of_none_cells_per_notebook.append(
            number_of_none_cells / number_of_code_cells
        )
        percentage_of_setup_cells_per_notebook.append(
            number_of_setup_cells / number_of_code_cells
        )
        percentage_of_ingestion_cells_per_notebook.append(
            number_of_ingestion_cells / number_of_code_cells
        )
        percentage_of_validation_cells_per_notebook.append(
            number_of_validation_cells / number_of_code_cells
        )
        percentage_of_preprocessing_cells_per_notebook.append(
            number_of_preprocessing_cells / number_of_code_cells
        )
        percentage_of_training_cells_per_notebook.append(
            number_of_training_cells / number_of_code_cells
        )
        percentage_of_evaluation_cells_per_notebook.append(
            number_of_evaluation_cells / number_of_code_cells
        )
        percentage_of_visualization_cells_per_notebook.append(
            number_of_visualization_cells / number_of_code_cells
        )
        percentage_of_checkpoints_cells_per_notebook.append(
            number_of_checkpoints_cells / number_of_code_cells
        )
        percentage_of_post_development_cells_per_notebook.append(
            number_of_post_development_cells / number_of_code_cells
        )
    print(
        "Average percentage of none cells per notebook: ",
        sum(percentage_of_none_cells_per_notebook)
        / len(percentage_of_none_cells_per_notebook),
    )
    print(
        "Average percentage of setup cells per notebook: ",
        sum(percentage_of_setup_cells_per_notebook)
        / len(percentage_of_setup_cells_per_notebook),
    )
    print(
        "Average percentage of ingestion cells per notebook: ",
        sum(percentage_of_ingestion_cells_per_notebook)
        / len(percentage_of_ingestion_cells_per_notebook),
    )
    print(
        "Average percentage of validation cells per notebook: ",
        sum(percentage_of_validation_cells_per_notebook)
        / len(percentage_of_validation_cells_per_notebook),
    )
    print(
        "Average percentage of preprocessing cells per notebook: ",
        sum(percentage_of_preprocessing_cells_per_notebook)
        / len(percentage_of_preprocessing_cells_per_notebook),
    )
    print(
        "Average percentage of training cells per notebook: ",
        sum(percentage_of_training_cells_per_notebook)
        / len(percentage_of_training_cells_per_notebook),
    )
    print(
        "Average percentage of evaluation cells per notebook: ",
        sum(percentage_of_evaluation_cells_per_notebook)
        / len(percentage_of_evaluation_cells_per_notebook),
    )
    print(
        "Average percentage of visualization cells per notebook: ",
        sum(percentage_of_visualization_cells_per_notebook)
        / len(percentage_of_visualization_cells_per_notebook),
    )
    print(
        "Average percentage of checkpoints cells per notebook: ",
        sum(percentage_of_checkpoints_cells_per_notebook)
        / len(percentage_of_checkpoints_cells_per_notebook),
    )
    print(
        "Average percentage of post development cells per notebook: ",
        sum(percentage_of_post_development_cells_per_notebook)
        / len(percentage_of_post_development_cells_per_notebook),
    )
    print("---" * 30)
    print(
        "Average percentage of none cells global: ",
        number_of_none_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of setup cells global: ",
        number_of_setup_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of ingestion cells global: ",
        number_of_ingestion_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of validation cells global: ",
        number_of_validation_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of preprocessing cells global: ",
        number_of_preprocessing_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of training cells global: ",
        number_of_training_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of evaluation cells global: ",
        number_of_evaluation_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of visualization cells global: ",
        number_of_visualization_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of checkpoints cells global: ",
        number_of_checkpoints_cells_global / number_of_code_cells_global,
    )
    print(
        "Average percentage of post development cells global: ",
        number_of_post_development_cells_global / number_of_code_cells_global,
    )


def create_headergen_csvs():
    with open(
        PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/evaluation_file.json",
        "r",
    ) as f:
        json_long = json.load(f)
    for tag in ALL_TAGS.keys():
        df = pd.DataFrame(columns=["content", "output_type", "tag"])
        for cell in json_long["source"]:
            if tag in cell["correct_tag_ours"]:
                df.loc[len(df.index)] = [
                    " ".join(cell["content_processed"]),
                    "not_existent",
                    1,
                ]
            else:
                df.loc[len(df.index)] = [
                    " ".join(cell["content_processed"]),
                    "not_existent",
                    0,
                ]
        with open(
            PATH_TO_HEADERGEN_EVALUATION_DIR
            + "eval/csvs/"
            + tag.replace("-", "_").replace(" ", "_")
            + "_headergen.csv",
            "w",
        ) as f:
            df.to_csv(f, index=False)


def test_on_final_validation_notebooks_hybrid_macro(regex: str):
    load_pre_trained_models()
    print("Testing on final validation ...")
    with open("final_validation_results_macro.txt", "w") as file:
        file.writelines("#############################\n" + regex + "\n")
    MACRO_AVERAGE_F1_SCORE_TOTAL = []
    MACRO_AVERAGE_PRECISION_SCORE_TOTAL = []
    MACRO_ACCURACY_TOTAL = []
    MACRO_AVERAGE_RECALL_SCORE_TOTAL = []
    AVERAGE_RECALL_POSITIVE = []
    AVERAGE_F1_POSITIVE = []
    AVERAGE_PRECISION_POSITIVE = []
    for tag in ALL_TAGS.keys():
        if (
            tag != ACTIVITY.SETUP_NOTEBOOK
            and tag != ACTIVITY.CHECK_RESULTS
            and tag != ACTIVITY.VISUALIZE_DATA
        ):
            print("Starting with " + tag + " ...")

            classifier = xgb.XGBClassifier()
            classifier.load_model(
                PATH_TO_MODELS_DIR
                + "model_"
                + tag.replace("-", "_").replace(" ", "_")
                + "_boost.json"
            )
            vectorizer = joblib.load(
                PATH_TO_MODELS_DIR
                + "vectorizer_"
                + tag.replace("-", "_").replace(" ", "_")
                + "_boost.joblib"
            )

            df = pd.read_csv(
                PATH_TO_VALIDATION_DIR
                + "/csvs/"
                + tag.replace("-", "_").replace(" ", "_")
                + "_final.csv"
            )
            print("Shape before dropping duplicates: " + str(df.shape))
            df = df.drop_duplicates(subset=["original_content"], keep="first")
            print("Shape after dropping duplicates: " + str(df.shape))
            x = df["content"].values
            y = df["tag"].values
            x = vectorizer.transform(x)
            y_pred = classifier.predict(x)

            # Add the predictions to the DataFrame
            df["y_pred"] = y_pred

            # Create a DataFrame that only contains instances where the predicted and actual values do not match
            df_errors = df[df["tag"] != df["y_pred"]]
            df_errors["original_content"] = df_errors["original_content"].apply(
                lambda x: x.replace("\n", " ")
            )

            # Print the rows where the prediction was incorrect
            df_errors.to_csv(
                PATH_TO_VALIDATION_DIR
                + "/errors/"
                + tag.replace(" ", "_")
                + "_final_validation_errors.csv",
                index=False,
            )

            print(classification_report(y, y_pred))
            report = classification_report(y, y_pred, output_dict=True)
            MACRO_AVERAGE_F1_SCORE_TOTAL.append(report["macro avg"]["f1-score"])
            MACRO_ACCURACY_TOTAL.append(report["accuracy"])
            MACRO_AVERAGE_PRECISION_SCORE_TOTAL.append(report["macro avg"]["precision"])
            MACRO_AVERAGE_RECALL_SCORE_TOTAL.append(report["macro avg"]["recall"])
            if "1" in report:
                AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
                AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
                AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
                positive_report_per_phase(report, "recall")
                positive_report_per_phase(report, "precision")
                positive_report_per_phase(report, "f1-score")
                with open("final_validation_results_macro.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                        + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        + "Positive F1: "
                        + str(report["1"]["f1-score"])
                        + "\n"
                        + "Positive Precision: "
                        + str(report["1"]["precision"])
                        + "\n"
                        + "Positive Recall: "
                        + str(report["1"]["recall"])
                        + "\n"
                        + "Average macro f1 score total: "
                        + str(
                            sum(MACRO_AVERAGE_F1_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_F1_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average accuracy score total: "
                        + str(sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL))
                        + "\n"
                        + "Average macro recall score total: "
                        + str(
                            sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average macro precision score total: "
                        + str(
                            sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                        )
                        + "\n"
                        + "###############################################\n"
                    )
                print("###############################################")
            else:
                with open("final_validation_results_macro.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                        + "Average macro f1 score total: "
                        + str(
                            sum(MACRO_AVERAGE_F1_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_F1_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average accuracy score total: "
                        + str(sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL))
                        + "\n"
                        + "Average macro recall score total: "
                        + str(
                            sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average macro precision score total: "
                        + str(
                            sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                        )
                        + "\n"
                    )
            print_macro_report_per_phase(report, "recall")
            print_macro_report_per_phase(report, "precision")
            print_macro_report_per_phase(report, "f1-score")
            print("Average accuracy score per phase: " + str(report["accuracy"]))
        else:
            print("Starting with " + tag + " ...")
            df = pd.read_csv(
                PATH_TO_VALIDATION_DIR
                + "/csvs/"
                + tag.replace("-", "_").replace(" ", "_")
                + "_final.csv"
            )

            df = df.drop_duplicates(subset=["original_content"], keep="first")
            x = df["content"].values
            y = df["tag"].values
            z = df["output_type"].values
            data = zip(x, y, z)
            y_pred = []
            for curr_content, curr_tag, curr_output_type in data:
                if tag == ACTIVITY.SETUP_NOTEBOOK:
                    y_pred.append(setup_phase_heuristic_hybrid(curr_content))
                elif tag == ACTIVITY.CHECK_RESULTS:
                    y_pred.append(checkpoint_activity_heuristic_hybrid(curr_content))
                elif tag == ACTIVITY.VISUALIZE_DATA:
                    y_pred.append(
                        data_visualization_heuristic_hybrid(
                            curr_content, curr_output_type
                        )
                    )

            # Add the predictions to the DataFrame
            df["y_pred"] = y_pred

            # Create a DataFrame that only contains instances where the predicted and actual values do not match
            df_errors = df[df["tag"] != df["y_pred"]]
            df_errors["original_content"] = df_errors["original_content"].apply(
                lambda x: x.replace("\n", " ")
            )

            # Print the rows where the prediction was incorrect
            df_errors.to_csv(
                PATH_TO_VALIDATION_DIR
                + "/errors/"
                + tag.replace(" ", "_")
                + "_errors_hybrid.csv",
                index=False,
            )

            report = classification_report(y, y_pred, output_dict=True)
            MACRO_ACCURACY_TOTAL.append(report["accuracy"])
            MACRO_AVERAGE_F1_SCORE_TOTAL.append(report["macro avg"]["f1-score"])
            MACRO_AVERAGE_PRECISION_SCORE_TOTAL.append(report["macro avg"]["precision"])
            MACRO_AVERAGE_RECALL_SCORE_TOTAL.append(report["macro avg"]["recall"])
            if "1" in report.keys():
                AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
                AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
                AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
                positive_report_per_phase(report, "recall")
                positive_report_per_phase(report, "precision")
                positive_report_per_phase(report, "f1-score")
                with open("final_validation_results_macro.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                        + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        + "Positive F1: "
                        + str(report["1"]["f1-score"])
                        + "\n"
                        + "Positive Precision: "
                        + str(report["1"]["precision"])
                        + "\n"
                        + "Positive Recall: "
                        + str(report["1"]["recall"])
                        + "\n"
                        + "Average macro f1 score total: "
                        + str(
                            sum(MACRO_AVERAGE_F1_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_F1_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average accuracy score total: "
                        + str(sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL))
                        + "\n"
                        + "Average macro recall score total: "
                        + str(
                            sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average macro precision score total: "
                        + str(
                            sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                        )
                        + "\n"
                    )
                print("###############################################")
            else:
                with open("final_validation_results_macro.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                        + "Average macro f1 score total: "
                        + str(
                            sum(MACRO_AVERAGE_F1_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_F1_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average accuracy score total: "
                        + str(sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL))
                        + "\n"
                        + "Average macro recall score total: "
                        + str(
                            sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                        )
                        + "\n"
                        + "Average macro precision score total: "
                        + str(
                            sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                            / len(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                        )
                        + "\n"
                    )
            print_macro_report_per_phase(report, "recall")
            print_macro_report_per_phase(report, "precision")
            print_macro_report_per_phase(report, "f1-score")
            print("Average accuracy score per phase: " + str(report["accuracy"]))
            print("###############################################")
    print("###############################################")
    print_total_average_macro_report(MACRO_ACCURACY_TOTAL, "Accuracy")
    print_total_average_macro_report(MACRO_AVERAGE_F1_SCORE_TOTAL, "macro F1")
    print_total_average_macro_report(
        MACRO_AVERAGE_PRECISION_SCORE_TOTAL, "macro Precision"
    )
    print_total_positive_report(AVERAGE_PRECISION_POSITIVE, "Precision")
    print_total_positive_report(AVERAGE_RECALL_POSITIVE, "Recall")
    print_total_positive_report(AVERAGE_F1_POSITIVE, "F1")
    with open("final_validation_results_macro.txt", "a") as f:
        f.write(
            "\n"
            + "AVERAGE MACRO F1 SCORE TOTAL: "
            + str(sum(MACRO_AVERAGE_F1_SCORE_TOTAL) / len(MACRO_AVERAGE_F1_SCORE_TOTAL))
            + "\n"
            + "AVERAGE MACRO PRECISION SCORE TOTAL: "
            + str(
                sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
                / len(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
            )
            + "\n"
            + "AVERAGE MACRO ACCURACY SCORE TOTAL: "
            + str(sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL))
            + "\n"
            + "AVERAGE MACRO RECALL SCORE TOAL: "
            + str(
                sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
                / len(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
            )
            + "\n"
            + "AVERAGE PRECISION POSITIVE: "
            + str(sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE))
            + "\n"
            + "AVERAGE RECALL POSITIVE: "
            + str(sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE))
            + "\n"
            + "AVERAGE F1 POSITIVE: "
            + str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE))
            + "\n"
            + "###############################################\n"
        )
        global F1_COMPLETE
        global PRECISION_COMPLETE
        global RECALL_COMPLETE
        global ACCURACY_COMPLETE
        F1_COMPLETE = sum(MACRO_AVERAGE_F1_SCORE_TOTAL) / len(
            MACRO_AVERAGE_F1_SCORE_TOTAL
        )
        PRECISION_COMPLETE = sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL) / len(
            MACRO_AVERAGE_PRECISION_SCORE_TOTAL
        )
        RECALL_COMPLETE = sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL) / len(
            MACRO_AVERAGE_RECALL_SCORE_TOTAL
        )
        ACCURACY_COMPLETE = sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL)

    return {
        "Average macro Recall": str(
            sum(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
            / len(MACRO_AVERAGE_RECALL_SCORE_TOTAL)
        ),
        "Average macro Precision": str(
            sum(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
            / len(MACRO_AVERAGE_PRECISION_SCORE_TOTAL)
        ),
        "Average macro F1": str(
            sum(MACRO_AVERAGE_F1_SCORE_TOTAL) / len(MACRO_AVERAGE_F1_SCORE_TOTAL)
        ),
        "Accuracy": str(sum(MACRO_ACCURACY_TOTAL) / len(MACRO_ACCURACY_TOTAL)),
        "Positive Recall": str(
            sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE)
        ),
        "Positive Precision": str(
            sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE)
        ),
        "Positive F1": str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE)),
    }


def test_on_headergen_notebooks_hybrid(regex: str):
    load_pre_trained_models()
    print("Testing on headergen data hybrid ...")
    with open("headergen_hybrid_results.txt", "w") as file:
        file.writelines("#############################\n" + regex + "\n")
    AVERAGE_MACRO_F1_SCORE_TOTAL = []
    AVERAGE_MACRO_PRECISION_SCORE_TOTAL = []
    AVERAGE_ACCURACY_TOTAL = []
    AVERAGE_MACRO_RECALL_SCORE_TOTAL = []
    AVERAGE_RECALL_POSITIVE = []
    AVERAGE_F1_POSITIVE = []
    AVERAGE_PRECISION_POSITIVE = []
    for tag in ALL_TAGS.keys():
        if (
            tag != ACTIVITY.SETUP_NOTEBOOK
            and tag != ACTIVITY.CHECK_RESULTS
            and tag != ACTIVITY.VISUALIZE_DATA
        ):
            print("Starting with " + tag + " ...")

            classifier = xgb.XGBClassifier()
            classifier.load_model(
                PATH_TO_MODELS_DIR
                + "model_"
                + tag.replace("-", "_").replace(" ", "_")
                + "_boost.json"
            )
            vectorizer = joblib.load(
                PATH_TO_MODELS_DIR
                + "vectorizer_"
                + tag.replace("-", "_").replace(" ", "_")
                + "_boost.joblib"
            )

            df = pd.read_csv(
                PATH_TO_HEADERGEN_EVALUATION_DIR
                + "/eval/csvs/"
                + tag.replace("-", "_").replace(" ", "_")
                + "_headergen.csv"
            )

            df = df.dropna()
            x = df["content"].values
            y = df["tag"].values
            x = vectorizer.transform(x)
            y_pred = classifier.predict(x)

            # Add the predictions to the DataFrame
            df["y_pred"] = y_pred

            # Create a DataFrame that only contains instances where the predicted and actual values do not match
            df_errors = df[df["tag"] != df["y_pred"]]
            df_errors["content"] = df_errors["content"].apply(
                lambda x: x.replace("\n", " ")
            )

            # Print the rows where the prediction was incorrect
            df_errors.to_csv(
                PATH_TO_HEADERGEN_EVALUATION_DIR
                + "eval/error_csvs/"
                + tag.replace(" ", "_")
                + "_headergen_errors_hybrid.csv",
                index=False,
            )

            print(classification_report(y, y_pred))
            report = classification_report(y, y_pred, output_dict=True)
            AVERAGE_MACRO_F1_SCORE_TOTAL.append(report["macro avg"]["f1-score"])
            AVERAGE_ACCURACY_TOTAL.append(report["accuracy"])
            AVERAGE_MACRO_PRECISION_SCORE_TOTAL.append(report["macro avg"]["precision"])
            AVERAGE_MACRO_RECALL_SCORE_TOTAL.append(report["macro avg"]["recall"])
            if "1" in report:
                AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
                AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
                AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
                positive_report_per_phase(report, "recall")
                positive_report_per_phase(report, "precision")
                positive_report_per_phase(report, "f1-score")
                with open("headergen_hybrid_results.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average Macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average Macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average Macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                        + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        + "Positive F1: "
                        + str(report["1"]["f1-score"])
                        + "\n"
                        + "Positive Precision: "
                        + str(report["1"]["precision"])
                        + "\n"
                        + "Positive Recall: "
                        + str(report["1"]["recall"])
                        + "\n"
                        + "###############################################\n"
                    )
                print("###############################################")
            else:
                with open("headergen_hybrid_results.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average Macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average Macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average Macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                    )
            print_macro_report_per_phase(report, "recall")
            print_macro_report_per_phase(report, "precision")
            print_macro_report_per_phase(report, "f1-score")
            print("Average accuracy score per phase: " + str(report["accuracy"]))
        else:
            print("Starting with " + tag + " ...")
            df = pd.read_csv(
                PATH_TO_HEADERGEN_EVALUATION_DIR
                + "/eval/csvs/"
                + ALL_TAGS[tag].replace("-", "_").replace(" ", "_")
                + "_headergen.csv"
            )

            df = df.dropna()
            x = df["content"].values
            y = df["tag"].values
            z = df["output_type"].values
            data = zip(x, y, z)
            y_pred = []
            if tag == ACTIVITY.SETUP_NOTEBOOK:
                for _, row in df.iterrows():
                    if setup_phase_heuristic_hybrid(str(row["content"])) != row["tag"]:
                        print(row["content"])
                        print(
                            "Predicted: "
                            + str(setup_phase_heuristic_hybrid(row["content"]))
                        )
                        print(row["tag"])
                        print("#######################")
            elif tag == ACTIVITY.CHECK_RESULTS:
                for _, row in df.iterrows():
                    if (
                        checkpoint_activity_heuristic_hybrid(str(row["content"]))
                        != row["tag"]
                    ):
                        print(row["content"])
                        print(
                            "Predicted: "
                            + str(checkpoint_activity_heuristic_hybrid(row["content"]))
                        )
                        print(row["tag"])
                        print("#######################")
            elif tag == ACTIVITY.VISUALIZE_DATA:
                for idx, row in df.iterrows():
                    if (
                        data_visualization_heuristic_hybrid(
                            str(row["content"]), str(row["output_type"])
                        )
                        != row["tag"]
                    ):
                        print(row["content"])
                        print(
                            "Predicted: "
                            + str(
                                data_visualization_heuristic_hybrid(
                                    row["content"], str(row["output_type"])
                                )
                            )
                        )
                        print(row["tag"])
                        print("#######################")
            for curr_content, _, curr_output_type in data:
                if tag == ACTIVITY.SETUP_NOTEBOOK:
                    y_pred.append(setup_phase_heuristic_hybrid(curr_content))
                elif tag == ACTIVITY.CHECK_RESULTS:
                    y_pred.append(checkpoint_activity_heuristic_hybrid(curr_content))
                elif tag == ACTIVITY.VISUALIZE_DATA:
                    y_pred.append(
                        data_visualization_heuristic_hybrid(
                            curr_content, curr_output_type
                        )
                    )

            # Add the predictions to the DataFrame
            df["y_pred"] = y_pred

            # Create a DataFrame that only contains instances where the predicted and actual values do not match
            df_errors = df[df["tag"] != df["y_pred"]]
            df_errors["content"] = df_errors["content"].apply(
                lambda x: x.replace("\n", " ")
            )

            # Print the rows where the prediction was incorrect
            df_errors.to_csv(
                PATH_TO_HEADERGEN_EVALUATION_DIR
                + "eval/error_csvs/"
                + tag.replace(" ", "_")
                + "_headergen_errors_hybrid.csv",
                index=False,
            )

            report = classification_report(y, y_pred, output_dict=True)
            AVERAGE_ACCURACY_TOTAL.append(report["accuracy"])
            AVERAGE_MACRO_F1_SCORE_TOTAL.append(report["macro avg"]["f1-score"])
            AVERAGE_MACRO_PRECISION_SCORE_TOTAL.append(report["macro avg"]["precision"])
            AVERAGE_MACRO_RECALL_SCORE_TOTAL.append(report["macro avg"]["recall"])
            if "1" in report.keys():
                AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
                AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
                AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
                positive_report_per_phase(report, "recall")
                positive_report_per_phase(report, "precision")
                positive_report_per_phase(report, "f1-score")
                with open("headergen_hybrid_results.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average Macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average Macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average Macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                        + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        + "Positive F1: "
                        + str(report["1"]["f1-score"])
                        + "\n"
                        + "Positive Precision: "
                        + str(report["1"]["precision"])
                        + "\n"
                        + "Positive Recall: "
                        + str(report["1"]["recall"])
                        + "\n"
                    )
                print("###############################################")
            else:
                with open("headergen_hybrid_results.txt", "a") as f:
                    f.write(
                        "###############################################\n"
                        + tag
                        + "\n"
                        + "Accuracy: "
                        + str(report["accuracy"])
                        + "\n"
                        + "Average Macro F1: "
                        + str(report["macro avg"]["f1-score"])
                        + "\n"
                        + "Average Macro Precision: "
                        + str(report["macro avg"]["precision"])
                        + "\n"
                        + "Average Macro Recall: "
                        + str(report["macro avg"]["recall"])
                        + "\n"
                    )
            print_macro_report_per_phase(report, "recall")
            print_macro_report_per_phase(report, "precision")
            print_macro_report_per_phase(report, "f1-score")
            print("Average accuracy score per phase: " + str(report["accuracy"]))
            print("###############################################")
    print("###############################################")
    print_total_average_macro_report(AVERAGE_ACCURACY_TOTAL, "Accuracy")
    print_total_average_macro_report(AVERAGE_MACRO_F1_SCORE_TOTAL, "Macro F1")
    print_total_average_macro_report(
        AVERAGE_MACRO_PRECISION_SCORE_TOTAL, "Macro Precision"
    )
    print_total_positive_report(AVERAGE_PRECISION_POSITIVE, "Precision")
    print_total_positive_report(AVERAGE_RECALL_POSITIVE, "Recall")
    print_total_positive_report(AVERAGE_F1_POSITIVE, "F1")
    with open("headergen_hybrid_results.txt", "a") as f:
        f.write(
            "\n"
            + "AVERAGE MACRO F1 SCORE TOTAL: "
            + str(sum(AVERAGE_MACRO_F1_SCORE_TOTAL) / len(AVERAGE_MACRO_F1_SCORE_TOTAL))
            + "\n"
            + "AVERAGE PRECISION SCORE TOTAL: "
            + str(
                sum(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
                / len(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
            )
            + "\n"
            + "AVERAGE ACCURACY SCORE TOTAL: "
            + str(sum(AVERAGE_ACCURACY_TOTAL) / len(AVERAGE_ACCURACY_TOTAL))
            + "\n"
            + "AVERAGE RECALL SCORE TOAL: "
            + str(
                sum(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
                / len(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
            )
            + "\n"
            + "AVERAGE PRECISION POSITIVE: "
            + str(sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE))
            + "\n"
            + "AVERAGE RECALL POSITIVE: "
            + str(sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE))
            + "\n"
            + "AVERAGE F1 POSITIVE: "
            + str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE))
            + "\n"
            + "###############################################\n"
        )
    return {
        "Average Macro Recall": str(
            sum(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
            / len(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
        ),
        "Average Macro Precision": str(
            sum(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
            / len(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
        ),
        "Average Macro F1": str(
            sum(AVERAGE_MACRO_F1_SCORE_TOTAL) / len(AVERAGE_MACRO_F1_SCORE_TOTAL)
        ),
        "Accuracy": str(sum(AVERAGE_ACCURACY_TOTAL) / len(AVERAGE_ACCURACY_TOTAL)),
        "Positive Recall": str(
            sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE)
        ),
        "Positive Precision": str(
            sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE)
        ),
        "Positive F1": str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE)),
    }


def assert_eval_json_is_equal():
    """Asserts the the content in the eval json is equal to the provided data by headergen in seperated files."""
    sum = 0
    with open(
        PATH_TO_HEADERGEN_EVALUATION_DIR + "/eval/source_of_truth_jupylabel.json",
        "r",
    ) as f:
        eval = json.load(f)

    for file in os.listdir(PATH_TO_HEADERGEN_EVALUATION_DIR + "headergen_ground_truth"):
        with open(
            PATH_TO_HEADERGEN_EVALUATION_DIR + "headergen_ground_truth/" + file,
            "r",
        ) as f:
            ground_truth = json.load(f)
            sum += len(ground_truth)
        with open(
            PATH_TO_HEADERGEN_EVALUATION_DIR
            + "eval/"
            + file.replace(".json", "")
            + "_concat_with_our_labels.json",
            "r",
        ) as f:
            eval_notebook = json.load(f)
        assert len(ground_truth) == len(eval_notebook["source"])

    with open(
        PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/source_of_truth_jupylabel.json",
        "r",
    ) as f:
        eval = json.load(f)
        assert sum == len(eval["source"])


def create_headergen_eval_json_raw():
    path_to_headergen_notebooks = (
        PATH_TO_HEADERGEN_EVALUATION_DIR + "notebooks_headergen_json/"
    )
    path_to_ground_truth = PATH_TO_HEADERGEN_EVALUATION_DIR + "headergen_ground_truth/"
    for i in range(len(os.listdir(path_to_headergen_notebooks))):
        file_name = os.listdir(path_to_headergen_notebooks)[i]
        print(os.listdir(path_to_headergen_notebooks)[i])
        assert (
            os.listdir(path_to_headergen_notebooks)[i]
            == os.listdir(path_to_ground_truth)[i]
        )
        with open(
            path_to_headergen_notebooks + os.listdir(path_to_headergen_notebooks)[i],
            "r",
        ) as f:
            notebook = json.load(f)

        with open(
            path_to_ground_truth + os.listdir(path_to_ground_truth)[i],
            "r",
        ) as f:
            ground_truth = json.load(f)
        assert len(notebook["cells"]) == len(ground_truth)
        new_json_file = {"source": []}
        for i in range(len(notebook["cells"])):
            create_new_concat_json = {
                "notebook_name": "",
                "content": [],
                "content_processed": [],
                "tag_pred": [],
                "correct_tag_ours": [],
                "headergen_tag": [],
                "headergen_sot": [],
            }
            if isinstance(notebook["cells"][i]["source"], str):
                notebook["cells"][i]["source"] = notebook["cells"][i]["source"].split(
                    "\n"
                )
            create_new_concat_json["notebook_name"] = file_name.replace(".json", "")
            create_new_concat_json["content"] = notebook["cells"][i]["source"]
            create_new_concat_json["content_processed"] = preprocess_source_cell_nlp(
                notebook["cells"][i]["source"]
            )
            key = str(i + 1)
            create_new_concat_json["headergen_tag"] = ground_truth[key]
            new_json_file["source"].append(create_new_concat_json)
        with open(
            PATH_TO_HEADERGEN_EVALUATION_DIR
            + "eval/"
            + file_name.replace(".json", "_concat_with_our_labels.json"),
            "w",
        ) as f:
            json.dump(new_json_file, f, indent=4, ensure_ascii=False)


def return_predictions(text_as_list, output_type):
    tags = []
    sentence = " ".join(text_as_list)

    if sentence == "":
        return ["None"]

    if setup_phase_heuristic_hybrid(sentence) == 1:
        tags.append(ACTIVITY.SETUP_NOTEBOOK)
    if data_visualization_heuristic_hybrid(sentence, output_type) == 1:
        tags.append(ACTIVITY.VISUALIZE_DATA)
    if checkpoint_activity_heuristic_hybrid(sentence) == 1:
        tags.append(ACTIVITY.CHECK_RESULTS)
    sentence = " ".join(text_as_list)
    for tag in EXPLICIT_MODELS:
        sentence_transformed = VECTORIZERS[tag].transform([sentence])
        prediciton = CLASSIFIERS[tag].predict(sentence_transformed)
        if prediciton == 1:
            tags.append(tag)
    if len(tags) == 0:
        tags.append("None")
    return tags


def insert_our_sot_labels_into_json():
    load_pre_trained_models()
    sum = 0
    assert_eval_json_is_equal()
    with open(
        PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/source_of_truth_jupylabel.json",
        "r",
    ) as f:
        eval = json.load(f)
    for file in os.listdir(PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/"):
        if file.endswith("_with_our_labels.json"):
            with open(
                PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/" + file,
                "r",
            ) as f:
                notebook = json.load(f)
                sum += len(notebook["source"])
            for cell in eval["source"]:
                for cell_new in notebook["source"]:
                    if (
                        cell["notebook_name"] == cell_new["notebook_name"]
                        and cell["content"] == cell_new["content"]
                    ):
                        cell_new["correct_tag_ours"] = cell["correct_tag_ours"]
                        cell_new["tag_pred"] = return_predictions(
                            cell["content_processed"], ""
                        )
            with open(
                PATH_TO_HEADERGEN_EVALUATION_DIR + "/eval/" + file,
                "w",
            ) as f:
                json.dump(notebook, f, indent=4, ensure_ascii=False)
    assert sum == len(eval["source"])


def insert_headergen_sot():
    for file in os.listdir(PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/"):
        if file.endswith("_our_labels.json"):
            with open(
                PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/" + file,
                "r",
            ) as f:
                notebook = json.load(f)
            with open(
                PATH_TO_HEADERGEN_EVALUATION_DIR
                + "headergen_ground_truth/"
                + file.replace("_concat_with_our_labels.json", ".json"),
                "r",
            ) as f:
                got = json.load(f)
            assert len(notebook["source"]) == len(got)
            for i in range(len(notebook["source"])):
                notebook["source"][i]["headergen_sot"] = got[str(i + 1)]
            with open(
                PATH_TO_HEADERGEN_EVALUATION_DIR + "/eval/" + file,
                "w",
            ) as f:
                json.dump(notebook, f, indent=4, ensure_ascii=False)


def create_big_eval():
    sum = 0
    big_json = {"source": []}
    for file in os.listdir(PATH_TO_HEADERGEN_EVALUATION_DIR + "/eval/"):
        if file.endswith("_our_labels.json"):
            with open(
                PATH_TO_HEADERGEN_EVALUATION_DIR + "/eval/" + file,
                "r",
            ) as f:
                notebook = json.load(f)
                sum += len(notebook["source"])
            for cell in notebook["source"]:
                big_json["source"].append(cell)
    assert len(big_json["source"]) == sum
    with open(
        PATH_TO_HEADERGEN_EVALUATION_DIR + "eval/evaluation_file.json",
        "w",
    ) as f:
        json.dump(big_json, f, indent=4, ensure_ascii=False)


def evaluate_after_executing_cli():
    load_pre_trained_models()
    with open("eval.txt", "w") as f:
        f.write("")
    with open("cli_run.json", "r") as f:
        cli = json.load(f)
    with open(
        PATH_TO_VALIDATION_DIR + "ML_DATA_HYBRID_FINAL_COMPLETE.json",
        "r",
    ) as f:
        sot = json.load(f)

    complete_f1_score = []
    complete_f1_score_2 = []
    complete_precision_score = []
    complete_precision_score_2 = []
    complete_recall_score = []
    complete_recall_score_2 = []
    complete_accuracy_score = []
    complete_accuracy_score_2 = []
    for tag in ALL_TAGS.keys():
        print("Starting with " + tag + " ...")
        classifier = xgb.XGBClassifier()
        classifier.load_model(
            PATH_TO_MODELS_DIR
            + "model_"
            + tag.replace("-", "_").replace(" ", "_")
            + "_boost.json"
        )
        vectorizer = joblib.load(
            PATH_TO_MODELS_DIR
            + "vectorizer_"
            + tag.replace("-", "_").replace(" ", "_")
            + "_boost.joblib"
        )
        assert len(cli["source"]) == len(sot["source"]), (
            "Length of cli was: "
            + str(len(cli["source"]))
            + " and length of sot was: "
            + str(len(sot["source"]))
        )
        df_cli = pd.DataFrame(
            columns=["content", "tag", "output_type", "original_content"]
        )
        df_sot = pd.DataFrame(
            columns=["content", "tag", "output_type", "original_content"]
        )
        df_explicit = pd.DataFrame(
            columns=["content", "tag", "output_type", "original_content"]
        )
        #######################################################################
        # Create the dataframes for cli and sot
        for cell in cli["source"]:
            assert cell["content"] == preprocess_source_cell_nlp(cell["content_old"])

            if tag in cell["tags"]:
                df_cli.loc[len(df_cli)] = (
                    " ".join(preprocess_source_cell_nlp(cell["content_old"])),
                    1,
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                )
            else:
                df_cli.loc[len(df_cli)] = (
                    " ".join(preprocess_source_cell_nlp(cell["content_old"])),
                    0,
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                )
        for cell in sot["source"]:
            if tag in cell["tags"]:
                df_sot.loc[len(df_sot)] = (
                    " ".join(preprocess_source_cell_nlp(cell["content_old"])),
                    1,
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                )
            else:
                df_sot.loc[len(df_sot)] = (
                    " ".join(preprocess_source_cell_nlp(cell["content_old"])),
                    0,
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                )
        #######################################################################
        # Create the dataframe for explicit predicitions
        for cell in cli["source"]:
            assert cell["content"] == preprocess_source_cell_nlp(cell["content_old"])
            if (
                tag == ACTIVITY.SETUP_NOTEBOOK
                or tag == ACTIVITY.VISUALIZE_DATA
                or tag == ACTIVITY.CHECK_RESULTS
            ):
                if tag == ACTIVITY.SETUP_NOTEBOOK:
                    df_explicit.loc[len(df_explicit)] = (
                        " ".join(cell["content"]),
                        setup_phase_heuristic_hybrid(" ".join(cell["content"])),
                        cell["output_type"],
                        " ".join(cell["content_old"]),
                    )
                    " ".join(cell["content_old"])
                elif tag == ACTIVITY.VISUALIZE_DATA:
                    df_explicit.loc[len(df_explicit)] = (
                        " ".join(cell["content"]),
                        data_visualization_heuristic_hybrid(
                            " ".join(cell["content"]), cell["output_type"]
                        ),
                        cell["output_type"],
                        " ".join(cell["content_old"]),
                    )
                elif tag == ACTIVITY.CHECK_RESULTS:
                    df_explicit.loc[len(df_explicit)] = (
                        " ".join(cell["content"]),
                        checkpoint_activity_heuristic_hybrid(" ".join(cell["content"])),
                        cell["output_type"],
                        " ".join(cell["content_old"]),
                    )
            else:
                df_explicit.loc[len(df_explicit)] = (
                    " ".join(cell["content"]),
                    classifier.predict(
                        vectorizer.transform(
                            [" ".join(preprocess_source_cell_nlp(cell["content_old"]))],
                        )
                    )[0],
                    cell["output_type"],
                    " ".join(cell["content_old"]),
                )
        #######################################################################
        print("Length of df_sot before dropping dups: " + str(len(df_sot)))
        print("Length of df_cli before dropping dups: " + str(len(df_cli)))
        print("Length of df_explicit before dropping dups: " + str(len(df_explicit)))
        assert len(df_cli) == len(df_explicit)
        print(df_cli.compare(df_explicit))
        diff = df_cli.compare(df_explicit)
        if len(diff) > 0:
            print(df_cli.loc[diff.index[0]])
            df_cli.loc[diff.index].to_csv("diff_" + tag + ".csv", index=False)
        #######################################################################
        # Check if contents, output_types and tags match for cli and explicit
        # Also check if contents and output_types match with sot
        print(df_cli.compare(df_explicit))
        for i in range(len(df_cli)):
            assert df_cli["content"].values[i] == df_explicit["content"].values[i]
            assert (
                df_cli["output_type"].values[i] == df_explicit["output_type"].values[i]
            )
            assert df_cli["tag"].values[i] == df_explicit["tag"].values[i], (
                "Problem with the following cell: "
                + str(df_cli["original_content"].values[i])
                + " and the following tag: "
                + tag
                + "output was for cli: "
                + str(df_cli["tag"].values[i])
                + " and for explicit: "
                + str(df_explicit["tag"].values[i])
            )
            assert df_cli["content"].values[i] == df_sot["content"].values[i]
            assert (
                df_cli["original_content"].values[i]
                == df_sot["original_content"].values[i]
            )
            assert df_cli["output_type"].values[i] == df_sot["output_type"].values[i], (
                "Problem with the following cell: "
                + str(df_cli["original_content"].values[i])
                + " and the following output type: "
                + str(df_cli["output_type"].values[i])
            )
        print("PASSED ALL EQUAL CHECKS!")
        #######################################################################
        df_cli = df_cli.drop_duplicates(subset=["original_content"], keep="first")
        df_sot = df_sot.drop_duplicates(subset=["original_content"], keep="first")
        df_explicit = df_explicit.drop_duplicates(
            subset=["original_content"], keep="first"
        )
        print("Length of df_sot after dropping dups: " + str(len(df_sot)))
        print("Length of df_explicit after dropping dups: " + str(len(df_explicit)))
        print("Length of df_cli after dropping dups: " + str(len(df_cli)))
        assert df_cli.equals(df_explicit)
        assert len(df_explicit) == len(df_cli)
        report = classification_report(
            df_sot["tag"].values, df_explicit["tag"].values, output_dict=True
        )
        report2 = classification_report(
            df_sot["tag"].values, df_cli["tag"].values, output_dict=True
        )
        complete_f1_score.append(report["macro avg"]["f1-score"])
        complete_f1_score_2.append(report2["macro avg"]["f1-score"])
        complete_precision_score.append(report["macro avg"]["precision"])
        complete_precision_score_2.append(report2["macro avg"]["precision"])
        complete_recall_score.append(report["macro avg"]["recall"])
        complete_recall_score_2.append(report2["macro avg"]["recall"])
        complete_accuracy_score.append(report["accuracy"])
        complete_accuracy_score_2.append(report2["accuracy"])
        with open("eval.txt", "a") as f:
            f.write(
                "###############################################\n"
                + "Tag: "
                + tag
                + "\n"
                + "F1 score: "
                + str(report["macro avg"]["f1-score"])
                + "\n"
                + "Recall score: "
                + str(report["macro avg"]["recall"])
                + "\n"
                + "Precision score: "
                + str(report["macro avg"]["precision"])
                + "\n"
                + "Accuracy score: "
                + str(report["accuracy"])
                + "\n"
                + "###############################################\n"
            )
        df_test_against_validation_csvs = pd.read_csv(
            PATH_TO_VALIDATION_DIR
            + "/csvs/"
            + tag.replace("-", "_").replace(" ", "_")
            + "_final.csv"
        )
        df_test_against_validation_csvs = (
            df_test_against_validation_csvs.drop_duplicates(
                subset=["original_content"], keep="first"
            )
        )
        df_test_against_validation_csvs = df_test_against_validation_csvs.sort_values(
            by="original_content"
        ).reset_index(drop=True)
        assert len(df_test_against_validation_csvs) == len(df_explicit)
        df_explicit = df_explicit.sort_values(by="original_content").reset_index(
            drop=True
        )
        df_sot = df_sot.sort_values(by="original_content").reset_index(drop=True)
        assert df_test_against_validation_csvs.equals(df_sot)
    global F1_COMPLETE
    global PRECISION_COMPLETE
    global RECALL_COMPLETE
    global ACCURACY_COMPLETE
    print("Complete f1 score: " + str(sum(complete_f1_score) / len(complete_f1_score)))
    print("Complete f1 score global: " + str(F1_COMPLETE))
    print(
        "Complete precision score: "
        + str(sum(complete_precision_score) / len(complete_precision_score))
    )
    print("Complete precision score global: " + str(PRECISION_COMPLETE))
    print(
        "Complete recall score: "
        + str(sum(complete_recall_score) / len(complete_recall_score))
    )
    print("Complete recall score global: " + str(RECALL_COMPLETE))
    print(
        "Complete accuracy score: "
        + str(sum(complete_accuracy_score) / len(complete_accuracy_score))
    )
    print("Complete accuracy score global: " + str(ACCURACY_COMPLETE))
    assert (sum(complete_f1_score) / len(complete_f1_score)) == (
        sum(complete_f1_score_2) / len(complete_f1_score_2)
    )
    assert (sum(complete_precision_score) / len(complete_precision_score)) == (
        sum(complete_precision_score_2) / len(complete_precision_score_2)
    )
    assert (sum(complete_recall_score) / len(complete_recall_score)) == (
        sum(complete_recall_score_2) / len(complete_recall_score_2)
    )
    assert (sum(complete_accuracy_score) / len(complete_accuracy_score)) == (
        sum(complete_accuracy_score_2) / len(complete_accuracy_score_2)
    )
    assert (sum(complete_f1_score) / len(complete_f1_score)) == F1_COMPLETE
    assert (
        sum(complete_precision_score) / len(complete_precision_score)
    ) == PRECISION_COMPLETE
    assert (sum(complete_recall_score) / len(complete_recall_score)) == RECALL_COMPLETE
    assert (
        sum(complete_accuracy_score) / len(complete_accuracy_score)
    ) == ACCURACY_COMPLETE
    print(
        "Passed all checks comparing results with validation notebook evaluation function"
    )


def update_ml_hybrid_files():
    update_pre_processing_in_ML()
    create_merged()
    update_final()


def new_preprocessing_training_and_validation_data():
    update_ml_hybrid_files()
    new_preprocessing_training()
    create_new_trainings_csvs()
    create_final_validation_csv()
    create_data_headergen()


def create_data_headergen():
    create_headergen_eval_json_raw()
    insert_our_sot_labels_into_json()
    insert_headergen_sot()
    create_big_eval()
    create_headergen_csvs()


def evaluate_on_headergen():
    create_data_headergen()
    return test_on_headergen_notebooks_hybrid(REGEX)


def big_evaluation():
    update_ml_hybrid_files()
    test_on_final_validation_notebooks_hybrid_macro(REGEX)
    evaluate_after_executing_cli()
