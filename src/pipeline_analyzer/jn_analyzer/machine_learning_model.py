import pandas as pd
import warnings
from pprint import pprint

warnings.filterwarnings("ignore")
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import joblib
from pipeline_analyzer.jn_analyzer.evaluation import (
    test_on_final_validation_notebooks_hybrid_macro,
    evaluate_on_headergen,
)
from pipeline_analyzer.jn_analyzer.constants import (
    REGEX,
    ALL_TAGS,
    ACTIVITY,
    PATH_TO_NEW_TRAINED_MODELS,
)
from sklearn.utils import resample


CONFIGS = {
    ACTIVITY.SETUP_NOTEBOOK: {
        "learning_rate": 0.29,
    },
    ACTIVITY.INGEST_DATA: {
        "learning_rate": 0.24,
    },
    ACTIVITY.PROCESS_DATA: {
        "learning_rate": 0.17,
    },
    ACTIVITY.TRAIN_MODEL: {
        "learning_rate": 0.16,
    },
    ACTIVITY.EVALUATE_MODEL: {
        "learning_rate": 0.26,
    },
    ACTIVITY.TRANSFER_RESULTS: {
        "learning_rate": 0.01,
    },
    ACTIVITY.VISUALIZE_DATA: {
        "learning_rate": 0.24,
    },
    ACTIVITY.VALIDATE_DATA: {
        "learning_rate": 0.21,
    },
}


def cv_model(word: str):
    config = {"n_jobs": -1, "objective": "binary:logistic"}
    grid_config = CONFIGS[word]
    print("##########################")
    print("Starting model for word: " + word)
    print("##########################")

    df3 = pd.read_csv(
        "./src/pipeline_analyzer/jn_analyzer/resources/training_data/training/csvs/"
        + word
        + "_shuffled.csv"
    )
    print("Number of samples with value 1: " + str(len(df3[df3.tag == 1])))
    print("Number of samples with value 0: " + str(len(df3[df3.tag == 0])))
    if len(df3[df3.tag == 1]) < len(df3[df3.tag == 0]):
        majority = df3[df3.tag == 0]
        minority = df3[df3.tag == 1]
    elif len(df3[df3.tag == 1]) > len(df3[df3.tag == 0]):
        majority = df3[df3.tag == 1]
        minority = df3[df3.tag == 0]

    df_majority = majority
    df_minority = minority

    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,  # sample with replacement
        n_samples=df_majority.shape[0],  # to match majority class
        random_state=123,
    )  # reproducible results

    # Combine majority class with upsampled minority class
    df3 = pd.concat([df_majority, df_minority_upsampled])

    train_sentences = df3["content"].values
    train_tags = df3["tag"].values

    x_train, x_test, y_train, y_test = train_test_split(
        train_sentences,
        train_tags,
        test_size=0.2,
        random_state=123,
        stratify=train_tags,
    )

    count_vectorizer = CountVectorizer(lowercase=False, token_pattern=REGEX)
    cv_train_vectors = count_vectorizer.fit_transform(x_train)
    cv_test_vectors = count_vectorizer.transform(x_test)
    classifier = XGBClassifier(
        learning_rate=grid_config["learning_rate"],
        objective=config["objective"],
        n_jobs=config["n_jobs"],
        verbosity=0,
        validate_parameters=True,
    )
    classifier.fit(
        cv_train_vectors,
        y_train,
    )
    y_test_pred = classifier.predict(cv_test_vectors)
    # Evaluate performance
    print(classification_report(y_test, y_test_pred))
    report = classification_report(y_test, y_test_pred, output_dict=True)
    pprint(report)
    joblib.dump(
        count_vectorizer,
        PATH_TO_NEW_TRAINED_MODELS
        + "vectorizer_"
        + word.replace(" ", "_").replace("-", "_")
        + "_boost.joblib",
    )

    classifier.save_model(
        PATH_TO_NEW_TRAINED_MODELS
        + "model_"
        + word.replace(" ", "_").replace("-", "_")
        + "_boost.json"
    )


def run_model():
    for word in ALL_TAGS.keys():
        cv_model(word)
    res_val_hybrid = test_on_final_validation_notebooks_hybrid_macro(REGEX)
    res_headergen_hybrid = evaluate_on_headergen()
    print("//////////////////////////////////")
    print("Result from val notebooks hybrid: ")
    pprint(res_val_hybrid)
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Result from headergen notebooks hybrid: ")
    pprint(res_headergen_hybrid)
