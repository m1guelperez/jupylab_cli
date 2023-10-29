# from pprint import pprint
# import math
# import numpy as np
# import json
# from pipeline_analyzer.jn_analyzer.utils import preprocess_source_cell_nlp
# import xgboost as xgb
# import os
# import joblib
# import pandas as pd
# from datetime import datetime
# from sklearn.metrics import classification_report
# import time

# F1_COMPLETE = None
# F1_COMPLETE_2 = None
# RECALL_COMPLETE = None
# RECALL_COMPLETE_2 = None
# PRECISION_COMPLETE = None
# PRECISION_COMPLETE_2 = None
# ACCURACY_COMPLETE = None
# ACCURACY_COMPLETE_2 = None

# words = [
#     "Data Pre-Processing Activity",
#     "Model Training Activity",
#     "Model Evaluation Activity",
#     "Data Ingestion Activity",
#     "Data Validation Activity",
#     "Setup Phase",
#     "Checkpoint Activity",
#     "Data Visualization Phase",
#     "Post Development Phase",
# ]


# TAG_MAPPINGS = {
#     "Setup Activity": "Setup_Activity",
#     "Data Ingestion": "Data_Ingestion_Activity",
#     "Data Validation": "Data_Validation_Activity",
#     "Data Pre-Processing": "Data_Pre_Processing_Activity",
#     "Model Training": "Model_Training_Activity",
#     "Model Evaluation": "Model_Evaluation_Activity",
#     "Checkpoint Activity": "Checkpoint_Activity",
#     "Post Development Phase": "Post_Development_Phase",
#     "Data Visualization Phase": "Data_Visualization_Phase",
# }

# TAG_MAPPINGS2 = {
#     "Setup Activity": "Setup_Activity",
#     "Data Ingestion Activity": "Data_Ingestion_Activity",
#     "Data Validation Activity": "Data_Validation_Activity",
#     "Data Pre-Processing Activity": "Data_Pre_Processing_Activity",
#     "Model Training Activity": "Model_Training_Activity",
#     "Model Evaluation Activity": "Model_Evaluation_Activity",
#     "Checkpoint Activity": "Checkpoint_Activity",
#     "Post Development Phase": "Post_Development_Phase",
#     "Data Visualization Phase": "Data_Visualization_Phase",
# }


# TAG_MAPPINGS_REVERSE = {
#     "Setup Activity": "Setup Activity",
#     "Data Ingestion": "Data Ingestion Activity",
#     "Data Validation": "Data Validation Activity",
#     "Data Pre-Processing": "Data Pre-Processing Activity",
#     "Model Training": "Model Training Activity",
#     "Model Evaluation": "Model Evaluation Activity",
#     "Checkpoint Activity": "Checkpoint Activity",
#     "Post Development Phase": "Post Development Phase",
#     "Data Visualization Phase": "Data Visualization Phase",
# }


# def print_positive_report(list_values, name):
#     print(
#         "Positive " + str(name) + " score: " + str(sum(list_values) / len(list_values))
#     )


# def print_total_average_weighted_report(list_values, name):
#     print(
#         "Average weighted "
#         + str(name)
#         + " score: "
#         + str(sum(list_values) / len(list_values))
#     )


# def print_total_average_macro_report(list_values, name):
#     print(
#         "Average macro "
#         + str(name)
#         + " score: "
#         + str(sum(list_values) / len(list_values))
#     )


# def print_weighted_report_per_phase(report, name):
#     print(
#         "Average weighted "
#         + name
#         + " score per phase: "
#         + str(report["weighted avg"][name.lower()])
#     )


# def print_macro_report_per_phase(report, name):
#     print(
#         "Average macro "
#         + name
#         + " score per phase: "
#         + str(report["macro avg"][name.lower()])
#     )


# def positive_report_per_phase(report, name):
#     print("Positive " + name + " score per phase: " + str(report["1"][name.lower()]))


# def print_total_positive_report(list_values, name):
#     print(
#         "Positive " + str(name) + " score: " + str(sum(list_values) / len(list_values))
#     )


# CLASSIFIERS = None
# VECTORIZERS = None


# def load_pre_trained():
#     global CLASSIFIERS
#     global VECTORIZERS
#     classifiers = {}
#     vectorizers = {}
#     for tag in TAG_MAPPINGS.keys():
#         print("Loading " + tag + " ...")
#         classifier = xgb.XGBClassifier()
#         classifier.load_model(
#             "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/model_"
#             + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             + "_boost-"
#             + str(datetime.now().strftime("%Y-%m-%d"))
#             + ".json"
#         )
#         classifiers[TAG_MAPPINGS[tag]] = classifier
#         vectorizers[TAG_MAPPINGS[tag]] = joblib.load(
#             "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/vectorizer_"
#             + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             + "_boost-"
#             + str(datetime.now().strftime("%Y-%m-%d"))
#             + ".joblib"
#         )

#     CLASSIFIERS = classifiers
#     VECTORIZERS = vectorizers


# def checkpoint_activity_heuristic_hybrid(content: str):
#     if "CHECKPOINT" in content:
#         return 1
#     else:
#         classifier = CLASSIFIERS[TAG_MAPPINGS["Checkpoint Activity"]]
#         vectorizer = VECTORIZERS[TAG_MAPPINGS["Checkpoint Activity"]]
#         vectorized_cell = vectorizer.transform([content])
#         prediction = classifier.predict(vectorized_cell)
#         if prediction == 1:
#             return 1
#         else:
#             return 0


# def data_visualization_heuristic_hybrid(content: str, output_type: str):
#     if output_type == "display_data":
#         return 1
#     else:
#         classifier = CLASSIFIERS[TAG_MAPPINGS["Data Visualization Phase"]]
#         vectorizer = VECTORIZERS[TAG_MAPPINGS["Data Visualization Phase"]]
#         vectorized_cell = vectorizer.transform([content])
#         prediction = classifier.predict(vectorized_cell)
#         if prediction == 1:
#             return 1
#         else:
#             return 0


# def setup_phase_heuristic_hybrid(content: str) -> int:
#     if "SETUP" in content:
#         return 1
#     else:
#         classifier = CLASSIFIERS[TAG_MAPPINGS["Setup Activity"]]
#         vectorizer = VECTORIZERS[TAG_MAPPINGS["Setup Activity"]]
#         vectorized_cell = vectorizer.transform([content])
#         prediction = classifier.predict(vectorized_cell)
#         if prediction[0] == 1:
#             return 1
#         else:
#             return 0


# def test_on_headergen_notebooks(regex: str):
#     """Average f1 score: 0.9735892086739943
#     Average precision score: 0.9516191068904051
#     Average accuracy score: 0.9739583333333334
#     Average recall score: 0.8850221713981101 with final model"""
#     print("Testing on headergen data ...")
#     with open("headergen_non_hybrid_results.txt", "w") as file:
#         file.writelines("#############################\n" + regex + "\n")
#     AVERAGE_WEIGHTED_F1_SCORE_TOTAL = []
#     AVERAGE_WEIGHTED_PRECISION_TOTAL = []
#     AVERAGE_ACCURACY_SCORE = []
#     AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL = []
#     AVERAGE_PRECISION_POSITIVE = []
#     AVERAGE_RECALL_POSITIVE = []
#     AVERAGE_F1_POSITIVE = []
#     for tag in TAG_MAPPINGS.keys():
#         print("Starting with " + tag + " ...")

#         classifier = xgb.XGBClassifier()
#         classifier.load_model(
#             "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/model_"
#             + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             + "_boost-"
#             + str(datetime.now().strftime("%Y-%m-%d"))
#             + ".json"
#         )
#         vectorizer = joblib.load(
#             "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/vectorizer_"
#             + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             + "_boost-"
#             + str(datetime.now().strftime("%Y-%m-%d"))
#             + ".joblib"
#         )

#         df = pd.read_csv(
#             "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/csvs/"
#             + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             + "_shuffled_b_concat.csv"
#         )

#         df = df.dropna()
#         x = df["content"].values
#         y = df["tag"].values
#         x = vectorizer.transform(x)
#         y_pred = classifier.predict(x)

#         # Add the predictions to the DataFrame
#         df["y_pred"] = y_pred

#         # Create a DataFrame that only contains instances where the predicted and actual values do not match
#         df_errors = df[df["tag"] != df["y_pred"]]

#         # Print the rows where the prediction was incorrect
#         df_errors.to_csv(
#             "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/errors_non_hybrid/"
#             + tag.replace(" ", "_")
#             + "_headergen_errors_non_hybrid.csv",
#             index=False,
#         )

#         print(classification_report(y, y_pred))
#         report = classification_report(y, y_pred, output_dict=True)
#         AVERAGE_WEIGHTED_F1_SCORE_TOTAL.append(report["weighted avg"]["f1-score"])
#         AVERAGE_ACCURACY_SCORE.append(report["accuracy"])
#         AVERAGE_WEIGHTED_PRECISION_TOTAL.append(report["weighted avg"]["precision"])
#         AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL.append(report["weighted avg"]["recall"])
#         if "1" in report:
#             AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
#             AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
#             AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
#             positive_report_per_phase(report, "recall")
#             positive_report_per_phase(report, "precision")
#             positive_report_per_phase(report, "f1-score")
#             with open("val_non_hybrid_results.txt", "a") as f:
#                 f.write(
#                     "###############################################\n"
#                     + tag
#                     + "\n"
#                     + "Accuracy: "
#                     + str(report["accuracy"])
#                     + "\n"
#                     + "Average Weighted F1: "
#                     + str(report["weighted avg"]["f1-score"])
#                     + "\n"
#                     + "Average Weighted Precision: "
#                     + str(report["weighted avg"]["precision"])
#                     + "\n"
#                     + "Average Weighted Recall: "
#                     + str(report["weighted avg"]["recall"])
#                     + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
#                     + "\n"
#                     + "Positive F1: "
#                     + str(report["1"]["f1-score"])
#                     + "\n"
#                     + "Positive Precision: "
#                     + str(report["1"]["precision"])
#                     + "\n"
#                     + "Positive Recall: "
#                     + str(report["1"]["recall"])
#                     + "\n"
#                 )
#             print("###############################################")
#         else:
#             with open("headergen_non_hybrid_results.txt", "a") as f:
#                 f.write(
#                     "###############################################\n"
#                     + tag
#                     + "\n"
#                     + "Accuracy: "
#                     + str(report["accuracy"])
#                     + "\n"
#                     + "Average Weighted F1: "
#                     + str(report["weighted avg"]["f1-score"])
#                     + "\n"
#                     + "Average Weighted Precision: "
#                     + str(report["weighted avg"]["precision"])
#                     + "\n"
#                     + "Average Weighted Recall: "
#                     + str(report["weighted avg"]["recall"])
#                     + "\n"
#                 )
#         print_weighted_report_per_phase(report, "recall")
#         print_weighted_report_per_phase(report, "precision")
#         print_weighted_report_per_phase(report, "f1-score")
#         print("Average accuracy score per phase: " + str(report["accuracy"]))

#     print("###############################################")
#     print_total_average_weighted_report(AVERAGE_WEIGHTED_F1_SCORE_TOTAL, "f1-score")
#     print_total_average_weighted_report(AVERAGE_WEIGHTED_PRECISION_TOTAL, "precision")
#     print_total_average_weighted_report(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL, "recall")
#     print_total_positive_report(AVERAGE_PRECISION_POSITIVE, "precision")
#     print_total_positive_report(AVERAGE_RECALL_POSITIVE, "recall")
#     print_total_positive_report(AVERAGE_F1_POSITIVE, "f1-score")

#     with open("headergen_non_hybrid_results.txt", "a") as f:
#         f.write(
#             "\n"
#             + "AVERAGE WEIGHTED F1 SCORE TOTAL: "
#             + str(
#                 sum(AVERAGE_WEIGHTED_F1_SCORE_TOTAL)
#                 / len(AVERAGE_WEIGHTED_F1_SCORE_TOTAL)
#             )
#             + "AVERAGE PRECISION SCORE TOTAL: "
#             + str(
#                 sum(AVERAGE_WEIGHTED_PRECISION_TOTAL)
#                 / len(AVERAGE_WEIGHTED_PRECISION_TOTAL)
#             )
#             + "\n"
#             + "AVERAGE ACCURACY SCORE TOTAL: "
#             + str(sum(AVERAGE_ACCURACY_SCORE) / len(AVERAGE_ACCURACY_SCORE))
#             + "\n"
#             + "AVERAGE RECALL SCORE TOAL: "
#             + str(
#                 sum(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL)
#                 / len(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL)
#             )
#             + "\n"
#             + "AVERAGE PRECISION POSITIVE: "
#             + str(sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE))
#             + "\n"
#             + "AVERAGE RECALL POSITIVE: "
#             + str(sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE))
#             + "\n"
#             + "AVERAGE F1 POSITIVE: "
#             + str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE))
#             + "\n"
#             + "###############################################\n"
#         )
#     return {
#         "Average Weighted Recall": sum(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL)
#         / len(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL),
#         "Average Weighted Precision": sum(AVERAGE_WEIGHTED_PRECISION_TOTAL)
#         / len(AVERAGE_WEIGHTED_PRECISION_TOTAL),
#         "Accuracy": sum(AVERAGE_ACCURACY_SCORE) / len(AVERAGE_ACCURACY_SCORE),
#         "Average Weighted F1": sum(AVERAGE_WEIGHTED_F1_SCORE_TOTAL)
#         / len(AVERAGE_WEIGHTED_F1_SCORE_TOTAL),
#         "Positive Recall": sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE),
#         "Positive Precision": sum(AVERAGE_PRECISION_POSITIVE)
#         / len(AVERAGE_PRECISION_POSITIVE),
#         "Positive F1": sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE),
#     }


# def test_on_headergen_notebooks_hybrid(regex: str):
#     load_pre_trained()
#     print("Testing on headergen data HYBRID ...")
#     with open("headergen_hybrid_results.txt", "w") as file:
#         file.writelines("#############################\n" + regex + "\n")
#     AVERAGE_MACRO_F1_SCORE_TOTAL = []
#     AVERAGE_MACRO_PRECISION_SCORE_TOTAL = []
#     AVERAGE_ACCURACY_TOTAL = []
#     AVERAGE_MACRO_RECALL_SCORE_TOTAL = []
#     AVERAGE_RECALL_POSITIVE = []
#     AVERAGE_F1_POSITIVE = []
#     AVERAGE_PRECISION_POSITIVE = []
#     for tag in TAG_MAPPINGS.keys():
#         if (
#             tag != "Setup Activity"
#             and tag != "Checkpoint Activity"
#             and tag != "Data Visualization Phase"
#         ):
#             print("Starting with " + tag + " ...")

#             classifier = xgb.XGBClassifier()
#             classifier.load_model(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/model_"
#                 + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#                 + "_boost-"
#                 + str(datetime.now().strftime("%Y-%m-%d"))
#                 + ".json"
#             )
#             vectorizer = joblib.load(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/vectorizer_"
#                 + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#                 + "_boost-"
#                 + str(datetime.now().strftime("%Y-%m-%d"))
#                 + ".joblib"
#             )

#             # df = pd.read_csv(
#             #     "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/csvs/"
#             #     + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             #     + "_shuffled_b_concat.csv"
#             # )

#             df = pd.read_csv(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/eval/own_csvs/"
#                 + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#                 + ".csv"
#             )

#             df = df.dropna()
#             x = df["content"].values
#             y = df["tag"].values
#             x = vectorizer.transform(x)
#             y_pred = classifier.predict(x)

#             # Add the predictions to the DataFrame
#             df["y_pred"] = y_pred

#             # Create a DataFrame that only contains instances where the predicted and actual values do not match
#             df_errors = df[df["tag"] != df["y_pred"]]
#             df_errors["content"] = df_errors["content"].apply(
#                 lambda x: x.replace("\n", " ")
#             )

#             # Print the rows where the prediction was incorrect
#             df_errors.to_csv(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/errors_hybrid/"
#                 + tag.replace(" ", "_")
#                 + "_headergen_errors_hybrid.csv",
#                 index=False,
#             )

#             print(classification_report(y, y_pred))
#             report = classification_report(y, y_pred, output_dict=True)
#             AVERAGE_MACRO_F1_SCORE_TOTAL.append(report["macro avg"]["f1-score"])
#             AVERAGE_ACCURACY_TOTAL.append(report["accuracy"])
#             AVERAGE_MACRO_PRECISION_SCORE_TOTAL.append(report["macro avg"]["precision"])
#             AVERAGE_MACRO_RECALL_SCORE_TOTAL.append(report["macro avg"]["recall"])
#             if "1" in report:
#                 AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
#                 AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
#                 AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
#                 positive_report_per_phase(report, "recall")
#                 positive_report_per_phase(report, "precision")
#                 positive_report_per_phase(report, "f1-score")
#                 with open("headergen_hybrid_results.txt", "a") as f:
#                     f.write(
#                         "###############################################\n"
#                         + tag
#                         + "\n"
#                         + "Accuracy: "
#                         + str(report["accuracy"])
#                         + "\n"
#                         + "Average Macro F1: "
#                         + str(report["macro avg"]["f1-score"])
#                         + "\n"
#                         + "Average Macro Precision: "
#                         + str(report["macro avg"]["precision"])
#                         + "\n"
#                         + "Average Macro Recall: "
#                         + str(report["macro avg"]["recall"])
#                         + "\n"
#                         + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
#                         + "Positive F1: "
#                         + str(report["1"]["f1-score"])
#                         + "\n"
#                         + "Positive Precision: "
#                         + str(report["1"]["precision"])
#                         + "\n"
#                         + "Positive Recall: "
#                         + str(report["1"]["recall"])
#                         + "\n"
#                         + "###############################################\n"
#                     )
#                 print("###############################################")
#             else:
#                 with open("headergen_hybrid_results.txt", "a") as f:
#                     f.write(
#                         "###############################################\n"
#                         + tag
#                         + "\n"
#                         + "Accuracy: "
#                         + str(report["accuracy"])
#                         + "\n"
#                         + "Average Macro F1: "
#                         + str(report["macro avg"]["f1-score"])
#                         + "\n"
#                         + "Average Macro Precision: "
#                         + str(report["macro avg"]["precision"])
#                         + "\n"
#                         + "Average Macro Recall: "
#                         + str(report["macro avg"]["recall"])
#                         + "\n"
#                     )
#             print_macro_report_per_phase(report, "recall")
#             print_macro_report_per_phase(report, "precision")
#             print_macro_report_per_phase(report, "f1-score")
#             print("Average accuracy score per phase: " + str(report["accuracy"]))
#         else:
#             print("Starting with " + tag + " ...")
#             # df = pd.read_csv(
#             #     "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/csvs/"
#             #     + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             #     + "_shuffled_b_concat.csv"
#             # )

#             df = pd.read_csv(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/eval/own_csvs/"
#                 + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#                 + ".csv"
#             )

#             df = df.dropna()
#             x = df["content"].values
#             y = df["tag"].values
#             z = df["output_type"].values
#             data = zip(x, y, z)
#             y_pred = []
#             if tag == "Setup Activity":
#                 for idx, row in df.iterrows():
#                     if setup_phase_heuristic_hybrid(str(row["content"])) != row["tag"]:
#                         print(row["content"])
#                         print(
#                             "Predicted: "
#                             + str(setup_phase_heuristic_hybrid(row["content"]))
#                         )
#                         print(row["tag"])
#                         print("#######################")
#             elif tag == "Checkpoint Activity":
#                 for idx, row in df.iterrows():
#                     if (
#                         checkpoint_activity_heuristic_hybrid(str(row["content"]))
#                         != row["tag"]
#                     ):
#                         print(row["content"])
#                         print(
#                             "Predicted: "
#                             + str(checkpoint_activity_heuristic_hybrid(row["content"]))
#                         )
#                         print(row["tag"])
#                         print("#######################")
#             elif tag == "Data Visualization Phase":
#                 for idx, row in df.iterrows():
#                     if (
#                         data_visualization_heuristic_hybrid(
#                             str(row["content"]), str(row["output_type"])
#                         )
#                         != row["tag"]
#                     ):
#                         print(row["content"])
#                         print(
#                             "Predicted: "
#                             + str(
#                                 data_visualization_heuristic_hybrid(
#                                     row["content"], str(row["output_type"])
#                                 )
#                             )
#                         )
#                         print(row["tag"])
#                         print("#######################")
#             for curr_content, curr_tag, curr_output_type in data:
#                 if tag == "Setup Activity":
#                     y_pred.append(setup_phase_heuristic_hybrid(curr_content))
#                 elif tag == "Checkpoint Activity":
#                     y_pred.append(checkpoint_activity_heuristic_hybrid(curr_content))
#                 elif tag == "Data Visualization Phase":
#                     y_pred.append(
#                         data_visualization_heuristic_hybrid(
#                             curr_content, curr_output_type
#                         )
#                     )

#             # Add the predictions to the DataFrame
#             df["y_pred"] = y_pred

#             # Create a DataFrame that only contains instances where the predicted and actual values do not match
#             df_errors = df[df["tag"] != df["y_pred"]]
#             df_errors["content"] = df_errors["content"].apply(
#                 lambda x: x.replace("\n", " ")
#             )

#             # Print the rows where the prediction was incorrect
#             df_errors.to_csv(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/errors_hybrid/"
#                 + tag.replace(" ", "_")
#                 + "_headergen_errors_hybrid.csv",
#                 index=False,
#             )

#             report = classification_report(y, y_pred, output_dict=True)
#             AVERAGE_ACCURACY_TOTAL.append(report["accuracy"])
#             AVERAGE_MACRO_F1_SCORE_TOTAL.append(report["macro avg"]["f1-score"])
#             AVERAGE_MACRO_PRECISION_SCORE_TOTAL.append(report["macro avg"]["precision"])
#             AVERAGE_MACRO_RECALL_SCORE_TOTAL.append(report["macro avg"]["recall"])
#             if "1" in report.keys():
#                 AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
#                 AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
#                 AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
#                 positive_report_per_phase(report, "recall")
#                 positive_report_per_phase(report, "precision")
#                 positive_report_per_phase(report, "f1-score")
#                 with open("headergen_hybrid_results.txt", "a") as f:
#                     f.write(
#                         "###############################################\n"
#                         + tag
#                         + "\n"
#                         + "Accuracy: "
#                         + str(report["accuracy"])
#                         + "\n"
#                         + "Average Macro F1: "
#                         + str(report["macro avg"]["f1-score"])
#                         + "\n"
#                         + "Average Macro Precision: "
#                         + str(report["macro avg"]["precision"])
#                         + "\n"
#                         + "Average Macro Recall: "
#                         + str(report["macro avg"]["recall"])
#                         + "\n"
#                         + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
#                         + "Positive F1: "
#                         + str(report["1"]["f1-score"])
#                         + "\n"
#                         + "Positive Precision: "
#                         + str(report["1"]["precision"])
#                         + "\n"
#                         + "Positive Recall: "
#                         + str(report["1"]["recall"])
#                         + "\n"
#                     )
#                 print("###############################################")
#             else:
#                 with open("headergen_hybrid_results.txt", "a") as f:
#                     f.write(
#                         "###############################################\n"
#                         + tag
#                         + "\n"
#                         + "Accuracy: "
#                         + str(report["accuracy"])
#                         + "\n"
#                         + "Average Macro F1: "
#                         + str(report["macro avg"]["f1-score"])
#                         + "\n"
#                         + "Average Macro Precision: "
#                         + str(report["macro avg"]["precision"])
#                         + "\n"
#                         + "Average Macro Recall: "
#                         + str(report["macro avg"]["recall"])
#                         + "\n"
#                     )
#             print_macro_report_per_phase(report, "recall")
#             print_macro_report_per_phase(report, "precision")
#             print_macro_report_per_phase(report, "f1-score")
#             print("Average accuracy score per phase: " + str(report["accuracy"]))
#             print("###############################################")
#     print("###############################################")
#     print_total_average_macro_report(AVERAGE_ACCURACY_TOTAL, "Accuracy")
#     print_total_average_macro_report(AVERAGE_MACRO_F1_SCORE_TOTAL, "Macro F1")
#     print_total_average_macro_report(
#         AVERAGE_MACRO_PRECISION_SCORE_TOTAL, "Macro Precision"
#     )
#     print_total_positive_report(AVERAGE_PRECISION_POSITIVE, "Precision")
#     print_total_positive_report(AVERAGE_RECALL_POSITIVE, "Recall")
#     print_total_positive_report(AVERAGE_F1_POSITIVE, "F1")
#     with open("headergen_hybrid_results.txt", "a") as f:
#         f.write(
#             "\n"
#             + "AVERAGE MACRO F1 SCORE TOTAL: "
#             + str(sum(AVERAGE_MACRO_F1_SCORE_TOTAL) / len(AVERAGE_MACRO_F1_SCORE_TOTAL))
#             + "\n"
#             + "AVERAGE PRECISION SCORE TOTAL: "
#             + str(
#                 sum(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
#                 / len(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
#             )
#             + "\n"
#             + "AVERAGE ACCURACY SCORE TOTAL: "
#             + str(sum(AVERAGE_ACCURACY_TOTAL) / len(AVERAGE_ACCURACY_TOTAL))
#             + "\n"
#             + "AVERAGE RECALL SCORE TOAL: "
#             + str(
#                 sum(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
#                 / len(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
#             )
#             + "\n"
#             + "AVERAGE PRECISION POSITIVE: "
#             + str(sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE))
#             + "\n"
#             + "AVERAGE RECALL POSITIVE: "
#             + str(sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE))
#             + "\n"
#             + "AVERAGE F1 POSITIVE: "
#             + str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE))
#             + "\n"
#             + "###############################################\n"
#         )
#     return {
#         "Average Macro Recall": str(
#             sum(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
#             / len(AVERAGE_MACRO_RECALL_SCORE_TOTAL)
#         ),
#         "Average Macro Precision": str(
#             sum(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
#             / len(AVERAGE_MACRO_PRECISION_SCORE_TOTAL)
#         ),
#         "Average Macro F1": str(
#             sum(AVERAGE_MACRO_F1_SCORE_TOTAL) / len(AVERAGE_MACRO_F1_SCORE_TOTAL)
#         ),
#         "Accuracy": str(sum(AVERAGE_ACCURACY_TOTAL) / len(AVERAGE_ACCURACY_TOTAL)),
#         "Positive Recall": str(
#             sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE)
#         ),
#         "Positive Precision": str(
#             sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE)
#         ),
#         "Positive F1": str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE)),
#     }


# def headergen_analysis():
#     markdown_counter = 0
#     code_cell_counter = 0
#     number_of_cells = 0
#     number_of_notebooks = 0
#     for file in os.listdir("./src/pipeline_analyzer/jn_analyzer/resources/notebooks/"):
#         if file.endswith(".ipynb"):
#             number_of_notebooks += 1
#             with open(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/notebooks/" + file, "r"
#             ) as f:
#                 json_content = json.load(f)
#                 for cell in json_content["cells"]:
#                     number_of_cells += 1
#                     if cell["cell_type"] == "markdown":
#                         markdown_counter += 1
#                     if cell["cell_type"] == "code":
#                         code_cell_counter += 1
#     print("Number of markdown cells: " + str(markdown_counter))
#     print("Number of code cells: " + str(code_cell_counter))
#     print("Number of cells in total: " + str(number_of_cells))
#     print("Number of notebooks: " + str(number_of_notebooks))
#     print(
#         "Average number of cells per notebook: "
#         + str(number_of_cells / number_of_notebooks)
#     )


# def create_new_validation_csvs():
#     """Uses our 120 evaluation notebooks"""
#     random_seed = 42
#     np.random.seed(random_seed)
#     print("Creating new validation csvs...")

#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/"
#         + str(datetime.now().strftime("%Y-%m-%d"))
#         + "_"
#         + "ML_DATA_HYBRID_FINAL_COMPLETE_NEW_PROCESSED.json",
#         "r",
#         encoding="utf-8",
#     ) as f:
#         file_as_json = json.load(f)
#     for tag in TAG_MAPPINGS:
#         df = pd.DataFrame(columns=["content", "tag", "output_type"])
#         index = 0
#         for cell in file_as_json["source"]:
#             if TAG_MAPPINGS_REVERSE[tag] in cell["tags"]:
#                 df.loc[index] = (" ".join(cell["content"]), 1, cell["output_type"])
#             else:
#                 df.loc[index] = (" ".join(cell["content"]), 0, cell["output_type"])
#             index += 1
#         if tag == "Model Evaluation":
#             df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
#             df.to_csv("Model.csv", index=False)
#         # remove nan rows
#         print("VALIDATION CSV for " + tag)
#         print("Before dropping nan rows: ", df.shape)
#         df = df.dropna()
#         print("After dropping nan rows: ", df.shape)

#         print("Before dropping dups: ", df.shape)
#         df = df.drop_duplicates(subset=["original_content"], keep="first")
#         print("After dropping dups: ", df.shape)
#         # drop empty strings
#         print("Before dropping empty strings: ", df.shape)
#         df = df[df["content"] != ""]
#         print("After dropping empty strings: ", df.shape)
#         # shuffle csv
#         df = df.sample(frac=1).reset_index(drop=True)
#         df.to_csv(
#             "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/csvs/"
#             + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#             + "_final.csv",
#             index=False,
#         )


# # create_new_validation_csvs()


# def create_headergen_json():
#     json_concat = {"content": []}
#     for file in os.listdir(
#         "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/notebooks/"
#     ):
#         if file.endswith(".json"):
#             with open(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/notebooks/"
#                 + file,
#                 "r",
#                 encoding="utf-8",
#             ) as f:
#                 file_as_json = json.load(f)
#                 for cell in file_as_json["cells"]:
#                     if cell["cell_type"] == "code":
#                         if len(cell["outputs"]) > 0:
#                             output_type = cell["outputs"][0]["output_type"]
#                         else:
#                             output_type = "not_existent"
#                         json_concat["content"].append(
#                             {
#                                 "source": (
#                                     lambda x: x if type(x) == list else x.split("\n")
#                                 )(cell["source"]),
#                                 "tags": [],
#                                 "output_type": output_type,
#                             }
#                         )
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/labeled_concat_test.json",
#         "w",
#     ) as f:
#         json.dump(json_concat, f, indent=4, ensure_ascii=False)


# # create_headergen_json()


# def new_preprocessing_training_and_validation_data():
#     update_ml_hybrid_files()
#     new_preprocessing_training()
#     create_new_trainings_csvs()
#     create_final_validation_csv()
#     preprocess_headergen_concat_json()
#     create_new_headergen_validation_csvs()


# def test_on_headergen_notebooks_hybrid_new():
#     load_pre_trained()
#     print("Testing on headergen data HYBRID NEW ...")

#     AVERAGE_WEIGHTED_F1_SCORE_TOTAL = []
#     AVERAGE_WEIGHTED_PRECISION_SCORE_TOTAL = []
#     AVERAGE_ACCURACY_TOTAL = []
#     AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL = []
#     AVERAGE_RECALL_POSITIVE = []
#     AVERAGE_F1_POSITIVE = []
#     AVERAGE_PRECISION_POSITIVE = []
#     for tag in TAG_MAPPINGS.keys():
#         if (
#             tag != "Setup Activity"
#             and tag != "Checkpoint Activity"
#             and tag != "Data Visualization Phase"
#         ):
#             print("Starting with " + tag + " ...")

#             classifier = CLASSIFIERS[TAG_MAPPINGS[tag]]
#             vectorizer = VECTORIZERS[TAG_MAPPINGS[tag]]

#             df = pd.read_csv(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/csvs/"
#                 + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#                 + "_shuffled_b_concat.csv"
#             )

#             df = df.dropna()
#             x = df["content"].values
#             y = df["tag"].values
#             x = vectorizer.transform(x)
#             y_pred = classifier.predict(x)

#             print(classification_report(y, y_pred))
#             report = classification_report(y, y_pred, output_dict=True)
#             AVERAGE_WEIGHTED_F1_SCORE_TOTAL.append(report["weighted avg"]["f1-score"])
#             AVERAGE_ACCURACY_TOTAL.append(report["accuracy"])
#             AVERAGE_WEIGHTED_PRECISION_SCORE_TOTAL.append(
#                 report["weighted avg"]["precision"]
#             )
#             AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL.append(report["weighted avg"]["recall"])
#             if "1" in report:
#                 AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
#                 AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
#                 AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
#                 positive_report_per_phase(report, "recall")
#                 positive_report_per_phase(report, "precision")
#                 positive_report_per_phase(report, "f1-score")
#                 print("###############################################")
#             print_weighted_report_per_phase(report, "recall")
#             print_weighted_report_per_phase(report, "precision")
#             print_weighted_report_per_phase(report, "f1-score")
#             print("Average accuracy score per phase: " + str(report["accuracy"]))
#         else:
#             print("Starting with " + tag + " ...")
#             df = pd.read_csv(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/csvs/"
#                 + TAG_MAPPINGS[tag].replace("-", "_").replace(" ", "_")
#                 + "_shuffled_b_concat.csv"
#             )

#             df = df.dropna()
#             x = df["content"].values
#             y = df["tag"].values
#             y_pred = []
#             for sentence in x:
#                 if tag == "Setup Activity":
#                     y_pred.append(setup_phase_heuristic_hybrid(sentence))
#                 elif tag == "Checkpoint Activity":
#                     y_pred.append(checkpoint_activity_heuristic_hybrid(sentence))

#             report = classification_report(y, y_pred, output_dict=True)
#             AVERAGE_ACCURACY_TOTAL.append(report["accuracy"])
#             AVERAGE_WEIGHTED_F1_SCORE_TOTAL.append(report["weighted avg"]["f1-score"])
#             AVERAGE_WEIGHTED_PRECISION_SCORE_TOTAL.append(
#                 report["weighted avg"]["precision"]
#             )
#             AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL.append(report["weighted avg"]["recall"])
#             if "1" in report.keys():
#                 AVERAGE_RECALL_POSITIVE.append(report["1"]["recall"])
#                 AVERAGE_F1_POSITIVE.append(report["1"]["f1-score"])
#                 AVERAGE_PRECISION_POSITIVE.append(report["1"]["precision"])
#                 positive_report_per_phase(report, "recall")
#                 positive_report_per_phase(report, "precision")
#                 positive_report_per_phase(report, "f1-score")
#                 print("###############################################")
#             print_weighted_report_per_phase(report, "recall")
#             print_weighted_report_per_phase(report, "precision")
#             print_weighted_report_per_phase(report, "f1-score")
#             print("Average accuracy score per phase: " + str(report["accuracy"]))
#             print("###############################################")
#     print("###############################################")
#     print_total_average_weighted_report(AVERAGE_ACCURACY_TOTAL, "Accuracy")
#     print_total_average_weighted_report(AVERAGE_WEIGHTED_F1_SCORE_TOTAL, "Weighted F1")
#     print_total_average_weighted_report(
#         AVERAGE_WEIGHTED_PRECISION_SCORE_TOTAL, "Weighted Precision"
#     )
#     print_total_positive_report(AVERAGE_PRECISION_POSITIVE, "Precision")
#     print_total_positive_report(AVERAGE_RECALL_POSITIVE, "Recall")
#     print_total_positive_report(AVERAGE_F1_POSITIVE, "F1")
#     return {
#         "Average Weighted Recall": str(
#             sum(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL)
#             / len(AVERAGE_WEIGHTED_RECALL_SCORE_TOTAL)
#         ),
#         "Average Weighted Precision": str(
#             sum(AVERAGE_WEIGHTED_PRECISION_SCORE_TOTAL)
#             / len(AVERAGE_WEIGHTED_PRECISION_SCORE_TOTAL)
#         ),
#         "Average Weighted F1": str(
#             sum(AVERAGE_WEIGHTED_F1_SCORE_TOTAL) / len(AVERAGE_WEIGHTED_F1_SCORE_TOTAL)
#         ),
#         "Accuracy": str(sum(AVERAGE_ACCURACY_TOTAL) / len(AVERAGE_ACCURACY_TOTAL)),
#         "Positive Recall": str(
#             sum(AVERAGE_RECALL_POSITIVE) / len(AVERAGE_RECALL_POSITIVE)
#         ),
#         "Positive Precision": str(
#             sum(AVERAGE_PRECISION_POSITIVE) / len(AVERAGE_PRECISION_POSITIVE)
#         ),
#         "Positive F1": str(sum(AVERAGE_F1_POSITIVE) / len(AVERAGE_F1_POSITIVE)),
#     }


# # test_on_headergen_notebooks_hybrid_new()


# def check_both_jsons():
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/labeled_concat.json"
#     ) as f:
#         sot = json.load(f)

#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/labeled_concat_test.json"
#     ) as f:
#         pre = json.load(f)

#     # check if the source keys are the same
#     print(len(sot["content"]))
#     print(len(pre["content"]))
#     assert len(sot["content"]) == len(pre["content"])
#     for cell in pre["content"]:
#         for cell_sot in sot["content"]:
#             if cell["source"] == cell_sot["source"]:
#                 cell["tags"] = cell_sot["tags"]
#                 break
#     print("Copying tags from sot to pre")

#     for cell in sot["content"]:
#         for cell_pre in pre["content"]:
#             if cell["source"] == cell_pre["source"]:
#                 cell_pre["tags"] = cell["tags"]
#                 break
#     # assert that both dictionaries have the same value for keys tags and source
#     for cell in pre["content"]:
#         for cell_sot in sot["content"]:
#             if cell["source"] == cell_sot["source"]:
#                 assert cell["tags"] == cell_sot["tags"]
#                 break
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/labeled_concat_test.json",
#         "w",
#     ) as f:
#         json.dump(pre, f, indent=4, ensure_ascii=False)


# def make_new_headergen_output_data():
#     create_headergen_json()
#     check_both_jsons()


# # make_new_headergen_output_data()


# def iterate_over_both_json():
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/concat_b1_b2_b3.json", "r"
#     ) as f:
#         concat = json.load(f)

#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/test_against_concat/ML_DATA_HYBRID_NEW.json",
#         "r",
#     ) as f:
#         ml = json.load(f)
#     print("Concat:")
#     print(len(concat["source"]))
#     print("ML:")
#     print(len(ml["source"]))
#     counter_equal = 0
#     counter_not_equal = 0
#     for cell in concat["source"]:
#         for cell_ml in ml["source"]:
#             if cell["content"] == cell_ml["content"]:
#                 if cell["tags"] == cell_ml["tags"]:
#                     counter_equal += 1
#                 else:
#                     counter_not_equal += 1
#                     pprint(cell["content"])
#                     print("//////////////////")
#                     pprint(cell_ml["content_old"])
#     print("Counter equal: " + str(counter_equal))
#     print("Counter not equal: " + str(counter_not_equal))


# # iterate_over_both_json()


# def update_pre_processing_in_ML():
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/validation_data/ML_DATA_HYBRID_FINAL_COMPLETE.json",
#         "r",
#     ) as f:
#         json_long = json.load(f)
#     for cell in json_long["source"]:
#         cell["content"] = preprocess_source_cell_nlp(cell["content_old"])
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/validation_data/ML_DATA_HYBRID_FINAL_COMPLETE.json",
#         "w",
#     ) as f:
#         json.dump(json_long, f, indent=4, ensure_ascii=False)

#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/validation_data/ML_DATA_HYBRID_FINAL.json",
#         "r",
#     ) as f:
#         json_long = json.load(f)
#     for cell in json_long["source"]:
#         cell["content"] = preprocess_source_cell_nlp(cell["content_old"])
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/validation_data/ML_DATA_HYBRID_FINAL.json",
#         "w",
#     ) as f:
#         json.dump(json_long, f, indent=4, ensure_ascii=False)


# def update_ml_hybrid_files():
#     update_pre_processing_in_ML()
#     create_merged()
#     update_final()


# def assert_notebooks_are_not_in_directory():
#     sample = []
#     for notebook in os.listdir(
#         "./src/pipeline_analyzer/jn_analyzer/resources/sample_of_notebooks_1"
#     ):
#         if notebook.endswith(".ipynb"):
#             sample.append(notebook)
#     for notebook in os.listdir(
#         "./src/pipeline_analyzer/jn_analyzer/resources/sample_of_notebooks_2"
#     ):
#         if notebook.endswith(".ipynb"):
#             sample.append(notebook)

#     for notebook in os.listdir(
#         "./src/pipeline_analyzer/jn_analyzer/resources/big_validation/validation_notebooks_120"
#     ):
#         if notebook.endswith(".ipynb"):
#             assert notebook not in sample
#     print(len(sample))
#     print(
#         len(
#             os.listdir(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/big_validation/validation_notebooks_120"
#             )
#         )
#     )
#     print("All notebooks are not in sample")


# # assert_notebooks_are_not_in_directory()


# def count_number_of_cells():
#     counter = 0
#     for notebook in os.listdir("./src/pipeline_analyzer/jn_analyzer/resources/inputs/"):
#         if notebook.endswith(".ipynb"):
#             with open(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/inputs/" + notebook
#             ) as f:
#                 data = json.load(f)
#                 cells = data["cells"]
#                 counter += len(cells)
#     print(counter / 1000)


# def analyse_parts_val():
#     number_of_code_cells = 0
#     number_of_none_cells = 0
#     number_of_setup_cells = 0
#     number_of_ingestion_cells = 0
#     number_of_validation_cells = 0
#     number_of_preprocessing_cells = 0
#     number_of_training_cells = 0
#     number_of_evaluation_cells = 0
#     number_of_visualization_cells = 0
#     number_of_checkpoints_cells = 0
#     number_of_post_development_cells = 0

#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_COMPLETE.json",
#         "r",
#         encoding="utf-8",
#     ) as f:
#         json_content = json.load(f)
#     number_of_code_cells = len(json_content["source"])
#     for cell in json_content["source"]:
#         if "Setup Activity" in cell["tags"]:
#             number_of_setup_cells += 1
#         if "Data Ingestion Activity" in cell["tags"]:
#             number_of_ingestion_cells += 1
#         if "Data Validation Activity" in cell["tags"]:
#             number_of_validation_cells += 1
#         if "Data Pre-Processing Activity" in cell["tags"]:
#             number_of_preprocessing_cells += 1
#         if "Model Training Activity" in cell["tags"]:
#             number_of_training_cells += 1
#         if "Model Evaluation Activity" in cell["tags"]:
#             number_of_evaluation_cells += 1
#         if "Data Visualization Phase" in cell["tags"]:
#             number_of_visualization_cells += 1
#         if "Checkpoint Activity" in cell["tags"]:
#             number_of_checkpoints_cells += 1
#         if "Post Development Phase" in cell["tags"]:
#             number_of_post_development_cells += 1
#         if "None" in cell["tags"] or len(cell["tags"]) == 0:
#             number_of_none_cells += 1

#     labels = (
#         "No Label\nAssigned",
#         "Setup\nActivity",
#         "Data\nIngestion",
#         "Data\nValidation",
#         "Data\nPre-Processing",
#         "Model\nTraining",
#         "Model\nEvaluation",
#         "Data\nVisualization",
#         "Checkpoint\nActivity",
#         "Post\nDevelopment Phase",
#     )
#     sizes = [
#         float(number_of_none_cells / number_of_code_cells),
#         float(number_of_setup_cells / number_of_code_cells),
#         float(number_of_ingestion_cells / number_of_code_cells),
#         float(number_of_validation_cells / number_of_code_cells),
#         float(number_of_preprocessing_cells / number_of_code_cells),
#         float(number_of_training_cells / number_of_code_cells),
#         float(number_of_evaluation_cells / number_of_code_cells),
#         float(number_of_visualization_cells / number_of_code_cells),
#         float(number_of_checkpoints_cells / number_of_code_cells),
#         float(number_of_post_development_cells / number_of_code_cells),
#     ]
#     print(
#         "Number of code cells: "
#         + str(number_of_code_cells)
#         + " and percentage: "
#         + str(number_of_code_cells / number_of_code_cells)
#     )
#     print(
#         "Number of setup cells: "
#         + str(number_of_setup_cells)
#         + " and percentage: "
#         + str(number_of_setup_cells / number_of_code_cells)
#     )
#     print(
#         "Number of ingestion cells: "
#         + str(number_of_ingestion_cells)
#         + " and percentage: "
#         + str(number_of_ingestion_cells / number_of_code_cells)
#     )
#     print(
#         "Number of validation cells: "
#         + str(number_of_validation_cells)
#         + " and percentage: "
#         + str(number_of_validation_cells / number_of_code_cells)
#     )
#     print(
#         "Number of preprocessing cells: "
#         + str(number_of_preprocessing_cells)
#         + " and percentage: "
#         + str(number_of_preprocessing_cells / number_of_code_cells)
#     )
#     print(
#         "Number of training cells: "
#         + str(number_of_training_cells)
#         + " and percentage: "
#         + str(number_of_training_cells / number_of_code_cells)
#     )
#     print(
#         "Number of evaluation cells: "
#         + str(number_of_evaluation_cells)
#         + " and percentage: "
#         + str(number_of_evaluation_cells / number_of_code_cells)
#     )
#     print(
#         "Number of visualization cells: "
#         + str(number_of_visualization_cells)
#         + " and percentage: "
#         + str(number_of_visualization_cells / number_of_code_cells)
#     )
#     print(
#         "Number of checkpoints cells: "
#         + str(number_of_checkpoints_cells)
#         + " and percentage: "
#         + str(number_of_checkpoints_cells / number_of_code_cells)
#     )
#     print(
#         "Number of post development cells: "
#         + str(number_of_post_development_cells)
#         + " and percentage: "
#         + str(number_of_post_development_cells / number_of_code_cells)
#     )
#     print(
#         "Number of none cells: "
#         + str(number_of_none_cells)
#         + " and percentage: "
#         + str(number_of_none_cells / number_of_code_cells)
#     )


# # analyse_parts_val()


# def get_file_size():
#     biggest_file = {"file": "", "size": -math.inf}
#     smallest_file = {"file": "", "size": math.inf}
#     average_file_size = 0
#     number_of_files = 0
#     for file in os.listdir(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/"
#     ):
#         file = (
#             "./src/pipeline_analyzer/jn_analyzer/resources/new_trained_models/" + file
#         )
#         if "model" in file:
#             average_file_size += os.path.getsize(file)
#             number_of_files += 1
#             # print(os.path.getsize(file))
#             if os.path.getsize(file) > biggest_file["size"]:
#                 biggest_file["file"] = file
#                 biggest_file["size"] = os.path.getsize(file)
#             if os.path.getsize(file) < smallest_file["size"]:
#                 smallest_file["file"] = file
#                 smallest_file["size"] = os.path.getsize(file)
#         if "vectorizer" in file:
#             average_file_size += os.path.getsize(file)
#             number_of_files += 1
#             print(os.path.getsize(file))
#             if os.path.getsize(file) > biggest_file["size"]:
#                 biggest_file["file"] = file
#                 biggest_file["size"] = os.path.getsize(file)
#             if os.path.getsize(file) < smallest_file["size"]:
#                 smallest_file["file"] = file
#                 smallest_file["size"] = os.path.getsize(file)
#     # convert to kilobytes
#     biggest_file["size"] = biggest_file["size"] / 1000
#     smallest_file["size"] = smallest_file["size"] / 1000
#     average_file_size = average_file_size / (number_of_files * 1000)
#     pprint(biggest_file)
#     pprint(smallest_file)
#     print(average_file_size)


# # get_file_size()


# def check_if_equal_with_sot():
#     """Check if both json have the same order and content"""
#     # Delete ALL kaggle cells
#     with open("test.json", "r") as f:
#         new_labeled = json.load(f)
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_COMPLETE.json",
#         "r",
#     ) as f:
#         sot = json.load(f)

#     counter = 0
#     counter2 = 0
#     print("Length of new: " + str(len(new_labeled["source"])))
#     print("Length of sot: " + str(len(sot["source"])))
#     all_sot_cells = []
#     all_new_cells = []
#     for cell in sot["source"]:
#         all_sot_cells.append(cell["content_old"])
#     for cell2 in new_labeled["source"]:
#         all_new_cells.append(cell2["content_old"])
#     for cell3 in all_sot_cells:
#         if cell3 not in all_new_cells:
#             print(cell3)
#             counter += 1
#     for cell3 in all_new_cells:
#         if cell3 not in all_sot_cells:
#             print(cell3)
#             counter2 += 1
#             break
#     assert counter == 0
#     assert counter2 == 0
#     print(counter)

#     for i in range(len(new_labeled["source"])):
#         assert (
#             new_labeled["source"][i]["content_old"] == sot["source"][i]["content_old"]
#         )
#         assert new_labeled["source"][i]["content"] == sot["source"][i]["content"]


# # check_if_equal_with_sot()


# def big_evaluation():
#     update_ml_hybrid_files()
#     test_on_final_validation_notebooks_hybrid_macro(r"(?u)[a-zA-Z]{1,}|[=[\]_]")
#     evaluate_after_executing_cli()


# def check_if_equal_with_sot():
#     with open("test.json", "r") as f:
#         cli = json.load(f)
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_COMPLETE.json",
#         "r",
#     ) as f:
#         sot = json.load(f)
#     with open("test_2.json", "r") as f:
#         cli_test_2 = json.load(f)
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_SHORT.json",
#         "r",
#     ) as f:
#         short = json.load(f)
#     print("Length of cli: " + str(len(cli["source"])))
#     print("Length of cli_test_2: " + str(len(cli_test_2["source"])))
#     list_of_contents = []
#     for cell in cli["source"]:
#         list_of_contents.append(cell["content_old"])
#     for cell in cli_test_2["source"]:
#         if cell["content_old"] not in list_of_contents:
#             print(cell["content_old"])

#     list_of_contents2 = []
#     for cell in cli_test_2["source"]:
#         list_of_contents2.append(cell["content_old"])
#     for cell in cli["source"]:
#         if cell["content_old"] not in list_of_contents2:
#             print(cell["content_old"])
#     assert len(short["source"]) == len(sot["source"])
#     assert len(cli_test_2["source"]) == len(cli["source"])
#     print("Passed length check")

#     # for i in range(len(cli_test_2["source"])):
#     #     assert (
#     #         short["source"][i]["content"] == sot["source"][i]["content_old"]
#     #     ), "Not equal"
#     #     assert (
#     #         short["source"][i]["output_type"] == sot["source"][i]["output_type"]
#     #     ), "Not equal"

#     # for i in range(len(short["source"])):
#     #     assert (
#     #         short["source"][i]["content"] == sot["source"][i]["content_old"]
#     #     ), "Not equal"
#     #     # assert (
#     #     #     short["source"][i]["output_type"] == sot["source"][i]["output_type"]
#     #     # ), "Not equal"
#     # print("Passed content check")
#     for i in range(len(cli_test_2["source"])):
#         assert (
#             short["source"][i]["content"] == sot["source"][i]["content_old"]
#         ), "Not equal"
#         assert cli_test_2["source"][i]["output_type"] == cli["source"][i]["output_type"]
#         cli["source"][i]["tags"] = cli_test_2["source"][i]["tags"]
#     with open("new_sot.json", "w") as f:
#         json.dump(cli, f, indent=4, ensure_ascii=False)


# # update_ml_hybrid_files()
# # check_if_equal_with_sot()
# def edit_cli_labeling_with_new_processing():
#     # new_preprocessing_training_and_validation_data()
#     load_pre_trained()
#     with open("test.json", "r") as f:
#         cli = json.load(f)
#     with open(
#         "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_SHORT.json",
#         "r",
#     ) as f:
#         sot = json.load(f)

#     with open("test_neu_.json", "r") as f:
#         cli_neu = json.load(f)

#     counter = 0
#     # for i in range(len(cli_neu["source"])):
#     #     # assert cli_neu["source"][i]["content_old"] == cli["source"][i]["content_old"]
#     #     assert cli_neu["source"][i]["content"] == cli["source"][i]["content"]
#     #     assert sot["source"][i]["content"] == cli["source"][i]["content_old"]

#     for i in range(len(cli["source"])):
#         cli["source"][i]["content"] = preprocess_source_cell_nlp(
#             cli["source"][i]["content_old"]
#         )
#         assert cli["source"][i]["content"] == preprocess_source_cell_nlp(
#             cli["source"][i]["content_old"]
#         )
#         tags = []
#         for tag in TAG_MAPPINGS2:
#             if (
#                 tag == "Setup Activity"
#                 or tag == "Data Visualization Phase"
#                 or tag == "Checkpoint Activity"
#             ):
#                 if tag == "Checkpoint Activity" and (
#                     checkpoint_activity_heuristic_hybrid(
#                         " ".join(cli["source"][i]["content"])
#                     )
#                     == 1
#                 ):
#                     tags.append(tag)
#                 if tag == "Data Visualization Phase" and (
#                     data_visualization_heuristic_hybrid(
#                         " ".join(cli["source"][i]["content"]),
#                         cli["source"][i]["output_type"],
#                     )
#                     == 1
#                 ):
#                     tags.append(tag)
#                 if tag == "Setup Activity" and (
#                     setup_phase_heuristic_hybrid(" ".join(cli["source"][i]["content"]))
#                     == 1
#                 ):
#                     tags.append(tag)
#             else:
#                 if (
#                     CLASSIFIERS[TAG_MAPPINGS2[tag]].predict(
#                         VECTORIZERS[TAG_MAPPINGS2[tag]].transform(
#                             [
#                                 " ".join(
#                                     preprocess_source_cell_nlp(
#                                         cli["source"][i]["content_old"]
#                                     )
#                                 )
#                             ]
#                         )
#                     )[0]
#                     == 1
#                 ):
#                     tags.append(tag)
#         if len(tags) == 0:
#             tags.append("None")
#         cli["source"][i]["tags"] = tags

#         # assert cli["content"] == preprocess_source_cell_nlp(cli["source"][i]["content_old"])

#     with open("test.json", "w") as f:
#         json.dump(cli, f, indent=4, ensure_ascii=False)


# # edit_cli_labeling_with_new_processing()
# # update_ml_hybrid_files()
# # evaluate_after_executing_cli()

# # big_evaluation()


# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_SHORT.json",
# #     "r",
# # ) as f:
# #     sot = json.load(f)
# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_SHORT_SELIN.json",
# #     "r",
# # ) as f:
# #     selin = json.load(f)

# # print("Length of sot: " + str(len(sot["source"])))
# # print("Length of selin: " + str(len(selin["source"])))

# # list_with_objects = []
# # for obj in selin["source"]:
# #     list_with_objects.append(obj["content"])

# # for i in range(len(sot["source"])):
# #     assert sot["source"][i]["content"] in list_with_objects, (
# #         "Content: " + str(sot["source"][i]["content"]) + " not in list"
# #     )

# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_SHORT.json",
# #     "r",
# # ) as f:
# #     sot = json.load(f)
# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/new_processing/final_validation/ML_DATA_HYBRID_FINAL_SHORT_SELIN.json",
# #     "r",
# # ) as f:
# #     selin = json.load(f)
# # print("Length of sot: " + str(len(sot["source"])))
# # print("Length of selin: " + str(len(selin["source"])))


# # for i in range(len(sot["source"])):
# #     assert sot["source"][i]["content"] == selin["source"][i]["content"]
# #     assert sot["source"][i]["output_type"] == selin["source"][i]["output_type"], (
# #         "Cell content was: "
# #         + str(sot["source"][i]["content"])
# #         + " and output type was: "
# #         + str(sot["source"][i]["output_type"])
# #     )

# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/eval/concat_big.json",
# #     "r",
# # ) as f:
# #     concat = json.load(f)
# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/eval/eval_bu.json",
# #     "r",
# # ) as f:
# #     eval2 = json.load(f)

# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/eval/eval.json",
# #     "r",
# # ) as f:
# #     eval = json.load(f)
# # print("Length of concat: " + str(len(concat["source"])))
# # print("Length of eval: " + str(len(eval["source"])))
# # for i in range(len(sot["content"])):
# #     assert sot["content"][i]["source"] == eval["source"][i]["content"]
# #     assert set(sot["content"][i]["tags"]) == set(
# #         eval["source"][i]["correct_tag_ours"]
# #     ), "Following cell content: " + str(sot["content"][i]["source"])

# # concat_content = []
# # eval_content = []
# # for cell in concat["source"]:
# #     concat_content.append(cell["content"])
# # for cell in eval["source"]:
# #     eval_content.append(cell["content"])

# # for cell in concat_content:
# #     assert cell in eval_content
# # new_big_json = {"source": []}
# # for cell in eval["source"]:
# #     json_obj = {
# #         "notebook_name": cell["notebook_name"],
# #         "content": cell["content"],
# #         "content_processed": cell["content_processed"],
# #         "correct_tag_ours": cell["correct_tag_ours"],
# #     }
# #     new_big_json["source"].append(json_obj)
# # with open(
# #     "./src/pipeline_analyzer/jn_analyzer/resources/headergen_notebook_json/eval/eval_new.json",
# #     "w",
# # ) as f:
# #     json.dump(new_big_json, f, indent=4, ensure_ascii=False)

import json
import os

# with open(
#     "./src/pipeline_analyzer/jn_analyzer/resources/training_data/training/json_processed_part1_b1.json"
# ) as f:
#     file1 = json.load(f)

# with open(
#     "./src/pipeline_analyzer/jn_analyzer/resources/training_data/training/json_processed_part1_b2.json"
# ) as f:
#     file2 = json.load(f)

# print("Length of file1: " + str(len(file1["content"])))
# print("Length of file2: " + str(len(file2["content"])))
# print("Sum of code cells: " + str(len(file1["content"]) + len(file2["content"])))
# print(
#     "Number of Notebooks: "
#     + str(
#         len(
#             os.listdir(
#                 "./src/pipeline_analyzer/jn_analyzer/resources/training_data/sample_of_notebooks"
#             )
#         )
#     )
# )
# NUMBER_OF_FILES = len(
#     os.listdir(
#         "./src/pipeline_analyzer/jn_analyzer/resources/training_data/sample_of_notebooks"
#     )
# )
# NUMBER_OF_CELLS = 0
# DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT = []
# for file in os.listdir(
#     "./src/pipeline_analyzer/jn_analyzer/resources/training_data/sample_of_notebooks"
# ):
#     try:
#         with open(
#             "./src/pipeline_analyzer/jn_analyzer/resources/training_data/sample_of_notebooks/"
#             + file
#         ) as f:
#             data = json.load(f)
#             if len(data["cells"]) == 0:
#                 continue
#             NUMBER_OF_CELLS += len(data["cells"])
#             number_of_cells_in_notebooks = 0
#             number_of_code_cells = 0
#             for cell in data["cells"]:
#                 if cell["cell_type"] == "code":
#                     number_of_code_cells += 1
#             DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT.append(
#                 number_of_code_cells / len(data["cells"])
#             )
#     except Exception as e:
#         print(e)
#         continue

# print("Total number of cells: " + str(NUMBER_OF_CELLS))
# print("Average number of cells per notebook: " + str(NUMBER_OF_CELLS / NUMBER_OF_FILES))
# print(
#     "Average percentage distribution of code cells per notebook: "
#     + str(
#         sum(DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT)
#         / len(DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT)
#     )
# )
# print(
#     "Average number of code cells per notebook: "
#     + str(
#         (
#             sum(DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT)
#             / len(DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT)
#         )
#         * (NUMBER_OF_CELLS / NUMBER_OF_FILES)
#     )
# )

# print(
#     "Used Notebooks: "
#     + str(
#         (len(file1["content"]) + len(file2["content"]))
#         / (
#             (
#                 sum(DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT)
#                 / len(DISTRIBUTION_OF_CODE_CELLS_PER_NOTEBOOK_PERCENT)
#             )
#             * (NUMBER_OF_CELLS / NUMBER_OF_FILES)
#         )
#     )
# )


with open(
    "./src/pipeline_analyzer/jn_analyzer/resources/validation_data/ML_DATA_HYBRID_FINAL_SHORT.json",
    "r",
) as f:
    new = json.load(f)
print(len(new["source"]))
# for i in range(len(old["source"])):
#     assert old["source"][i]["content"] == new["source"][i]["content"]
#     if "Setup Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.SETUP_NOTEBOOK in new["source"][i]["tags"]
#     if "Data Ingestion Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.INGEST_DATA in new["source"][i]["tags"]
#     if "Data Validation Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.VALIDATE_DATA in new["source"][i]["tags"]
#     if "Data Pre-Processing Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.PROCESS_DATA in new["source"][i]["tags"]
#     if "Model Training Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.TRAIN_MODEL in new["source"][i]["tags"]
#     if "Model Evaluation Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.EVALUATE_MODEL in new["source"][i]["tags"]
#     if "Checkpoint Activity" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.CHECK_RESULTS in new["source"][i]["tags"]
#     if "Data Visualization Phase" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.VISUALIZE_DATA in new["source"][i]["tags"]
#     if "Post Development Phase" in old["source"][i]["tags"]:
#         assert ACTIVITY_NAMES.TRANSFER_RESULTS in new["source"][i]["tags"]
# print("Done")
