from pprint import pprint
import re
from colorama import Fore, Style
from pipeline_analyzer.jn_analyzer.constants import (
    DEBUG_MODE,
    INSERT_HEADERS,
    CLASSIFIERS,
    VECTORIZERS,
    ALL_TAGS,
    EXPLICIT_MODELS,
    ACTIVITY,
)
import joblib
import xgboost as xgb
from datetime import datetime

OFFSET = 0


pattern_lhs = re.compile(r"^(?:(?:[A-Z_0-9]*[a-z_,\s]+)|(?:[a-z_]+[A-Z_0-9a-z\s]*))=")
pattern_slicing_assignment = re.compile(r"^[a-zA-Z]{1,}\[['\"a-zA-Z]{1,}\]\s?=")
slicing_pattern = re.compile(r"[a-zA-Z_]{1,}\[(?:[\[\]_0-9'\"a-zA-Z]|==){1,}\]")
pattern_delete_assignment_spaces = re.compile(r"(\s=\s|\s=|=\s)")
pattern_print = re.compile(r"\bprint\(.+\)")
pattern_print_replacement = re.compile(r"\"[^\"]{1,}\"")
pattern_paths = re.compile(r"(/([\w.-]+|\s?){1,}){1,}/?")
pattern_constants_without_suffix = re.compile(r"^[A-Z]{2,}(?:[_A-Z0-9]{1,})\s?=")


def path_trim_extension(path: str) -> str:
    counter = 0
    for char in reversed(path):
        if char != ".":
            counter += 1
        else:
            break
    return path[:-counter]


def delete_empty_strings_from_list(text_as_list: list) -> list:
    return [x for x in text_as_list if x]


def clear_prints(text_as_list: list) -> list:
    temp = []
    found_prints = False
    for line in text_as_list:
        if len(pattern_print.findall(line)) > 0:
            # removes only the strings INSIDE the print function and not the method calls
            # temp.append(pattern_string_in_print.sub("", line.replace(" ", "")))
            temp.append(pattern_print_replacement.sub("", line))
            found_prints = True
        else:
            temp.append(line)
    if found_prints:
        temp.insert(0, "CHECKPOINT")
    return temp


def delete_newline_character(text_as_list: list) -> list:
    return [str(x).replace("\n", "").strip() for x in text_as_list]


def delete_paths(text_as_list: list) -> list:
    temp = []
    # Some notebooks have their content not saved as list
    if isinstance(text_as_list, str):
        text_as_list = text_as_list.split("\n")
    for text in text_as_list:
        if len(pattern_paths.findall(text)) > 0:
            temp.append(pattern_paths.sub("path", text))
        else:
            temp.append(text)
    return temp


def delete_assingment_lhs(text_as_list: list) -> list:
    temp = []
    for text in text_as_list:
        text = pattern_delete_assignment_spaces.sub("=", text)
        # deletes all spaces around the = sign, this makes it easier for the following regex to match, since we do not have to consider spaces in the expression
        # Makes sure we are not splitting if statements using ==, >=, <= or !=
        if len(pattern_lhs.findall(text.strip())) > 0:
            temp.append(text.split("=")[1])
        else:
            temp.append(text)
    return temp


def clean_notebook_from_newlines(text_as_list: list) -> list:
    processed = delete_newline_character(text_as_list)
    processed = delete_empty_strings_from_list(processed)
    return processed


def get_number_of_words(text_as_list: list) -> int:
    temp = []
    for item in text_as_list:
        temp.append(re.split(" |,|\.|=", item.replace(" ", "")))
    return len(temp)


def clean_notebook_from_comments(text_as_list: list) -> list:
    """Removes all comments that are on top of a line e.g:

    #The next line prints the variable x

    print(x)

    but also deletes comments that are in the middle of a line e.g:

    print(x) #This prints the variable x
    """
    # TODO: Avoid double emptying of lists
    cleaned_from_empty_strings = [x for x in text_as_list if x]
    cleaned_top_comments = [x for x in cleaned_from_empty_strings if x[0] != "#"]
    removed_all_comments = []
    for x in cleaned_top_comments:
        found_quotes = False
        sentence_length = len(x)
        for char_pos, char in enumerate(x):
            if char == "#" and not found_quotes:
                removed_all_comments.append(x[:char_pos])
                break
            if char == '"' and found_quotes:
                found_quotes = False
            if char == '"' and not found_quotes:
                found_quotes = True
            if char_pos == sentence_length - 1:
                removed_all_comments.append(x)
    return removed_all_comments


def get_number_of_chars_from_list(text_as_list: list) -> int:
    return sum([len(x) for x in text_as_list])


def rename_assignments(text_as_list: list) -> list:
    new_list = []
    found_names = []
    for line in text_as_list:
        scanned_for_assignments = pattern_lhs.findall(line)
        # Ignore declarations of constants
        if len(scanned_for_assignments) > 0:
            if scanned_for_assignments[0] not in found_names:
                found_names.append(
                    str(scanned_for_assignments[0])
                    .replace("=", "")
                    .replace(" .", ".")
                    .strip()
                )
            new_list.append(
                re.sub(
                    r"\b%s\b"
                    % str(scanned_for_assignments[0]).replace("=", "").strip(),
                    "ASSIGN",
                    str(line).replace("  ", " "),
                )
            )
        else:
            if len(found_names) == 0:
                new_list.append(line)
            # We do not want to replace score with ASSIGN in ["scores = df['score']", "scores"]
            elif "=" in line or (
                not line.strip()[-1].isalpha() and not line.strip()[-1].isdigit()
            ):
                line_copy = None
                for name in found_names:
                    line_copy = re.sub(
                        r"\b%s\b" % name,
                        "ASSIGN",
                        str(line).replace("  ", " "),
                    )
                    line = line_copy
                new_list.append(line_copy)
            else:
                new_list.append(line)
    return new_list


def rename_const(text_as_list: str) -> list:
    new_list = []
    already_found = False
    for line in text_as_list:
        if (
            len(pattern_constants_without_suffix.findall(line)) > 0
            and not already_found
        ):
            new_list.insert(0, "SETUP")
            already_found = True
        elif len(pattern_constants_without_suffix.findall(line)) > 0 and already_found:
            continue
        else:
            new_list.append(line)
    return new_list


def rename_imports(text_as_list: str) -> list:
    """ "Deletes all imports and labels the cell with SETUP"""
    # We have to check for both, import and from at the beginning, otherwise we also accept plt.title("Import China") or if we omit the from check,
    # since we match from beginning we lose from tqdm import tqdm
    new_list = []
    already_renamed = False
    for line in text_as_list:
        if (
            len(re.findall(r"^\bimport\b", str(line).strip())) > 0
            or len(re.findall(r"^\bfrom\b", str(line).strip())) > 0
        ) and not already_renamed:
            new_list.insert(0, "SETUP")
            already_renamed = True
        elif (
            len(re.findall(r"^\bimport\b", str(line).strip())) > 0
            or len(re.findall(r"^\bfrom\b", str(line).strip())) > 0
        ) and already_renamed:
            continue
        else:
            new_list.append(line)
    return new_list


def rename_magic_commands(text_as_list: list) -> list:
    """Removes all magic commands from the notebook. If the cell contains a magic command and was not labeled as SETUP, it will be labeled as SETUP"""
    setup_cell = False
    for line in text_as_list:
        if "SETUP" in line:
            setup_cell = True
            break
    temp = []
    for line in text_as_list:
        if (
            str(line).strip().startswith("%") or str(line).strip().startswith("!")
        ) and setup_cell:
            continue
        elif (
            str(line).strip().startswith("%") or str(line).strip().startswith("!")
        ) and not setup_cell:
            temp.insert(0, "SETUP")
        else:
            temp.append(line)
    return temp


def rename_implicit_returns(text_as_list: list) -> list:
    temp = text_as_list
    if len(temp) == 0:
        return []
    elif (
        (str(temp[-1])[-1].isalpha() or str(temp[-1])[-1].isdigit())
        and "=" not in str(temp[-1])
        and (temp[-1] != "ASSIGN" and temp[-1] != "SETUP")
        and len(temp[-1].split(" ")) == 1
    ):
        temp.insert(0, "CHECKPOINT")
    return temp


def rename_slicing_assignments(text_as_list: list) -> list:
    """Replaces all slices if on the left side of an assignment was SLICE"""
    # Acts like a refined assignment analysis
    new_list = []
    for line in text_as_list:
        if len(pattern_slicing_assignment.findall(line)) > 0:
            new_list.append(
                re.sub(
                    slicing_pattern,
                    "SLICE",
                    line,
                )
            )
        else:
            new_list.append(line)
    return new_list


def preprocess_source_cell_nlp(text_as_list: list) -> list:
    if isinstance(text_as_list, str):
        text_as_list = text_as_list.split("\n")
    if len(text_as_list) == 0:
        return []
    processed = clean_notebook_from_comments(text_as_list)
    processed = clean_notebook_from_newlines(processed)
    processed = delete_paths(processed)
    processed = clear_prints(processed)
    processed = rename_const(processed)
    processed = rename_imports(processed)
    processed = rename_slicing_assignments(processed)
    processed = rename_magic_commands(processed)
    processed = rename_assignments(processed)
    processed = rename_implicit_returns(processed)
    return processed


def get_activity_hits(cell_dict: dict) -> list:
    hits = []
    if cell_dict["cell_type"] != "code":
        return hits
    for key, value in cell_dict["activities"].items():
        if float(value) > 0:
            hits.append(key)
    return hits


def insert_tags(phase: list, cell_number: int, original_notebook: dict):
    """Will either be a list or a string"""
    if INSERT_HEADERS[0] == False:
        original_notebook["cells"][cell_number]["metadata"]["tags"] = phase
    elif INSERT_HEADERS[0] and "None" not in phase:
        global OFFSET
        original_notebook["cells"].insert(
            cell_number + OFFSET,
            {
                "cell_type": "markdown",
                "metadata": {"position": cell_number + OFFSET},
                "source": ["##### " + ", ".join(phase)],
            },
        )
        OFFSET += 1


def make_labels(heuristics_dict: dict, original_notebook: dict):
    """This function takes the heuristics dictionary and the notebook and inserts the labels, sorted by value into the notebook"""
    global OFFSET
    OFFSET = 0
    for key in heuristics_dict:
        tags = []
        hits = get_activity_hits(heuristics_dict[key])  # maybe shift in if
        if heuristics_dict[key]["cell_type"] == "code":
            if len(hits) == 0:
                if DEBUG_MODE[0]:
                    print("No label found")
                    pprint(heuristics_dict[key])
                insert_tags(["None"], key, original_notebook)
            elif len(hits) > 0:
                for hit in hits:
                    tags.append(ALL_TAGS[hit])
                insert_tags(tags, key, original_notebook)


def create_recommendations(heuristics_dict: dict):
    for cell_number in heuristics_dict:
        if heuristics_dict[cell_number]["cell_type"] == "code":
            if heuristics_dict[cell_number]["number_of_tags"] > 1:
                print(
                    "The cell with the number "
                    + str(cell_number)
                    + " has more than one tag!\nPlease consider to split it into multiple cells!"
                )
            if heuristics_dict[cell_number]["number_of_tags"] == 0:
                print(
                    "Could not evaluate a tag for the cell with the number: "
                    + str(cell_number)
                    + "!\nPlease consider to refactor the cell, such that the intention gets clearer."
                )


def debug_print(text: str):
    if DEBUG_MODE[0]:
        print(text)


# Each print is seperated by a newline
def number_of_print_calls(text_as_list: list) -> int:
    counter = 0
    for line in text_as_list:
        if len(pattern_print.findall(line)) > 0:
            counter += 1
    return counter


def delete_kaggle_cells(analyzed_dict: dict) -> dict:
    """This method automatically removes kaggle cells from our evaluation set as well as a particular spanish cell."""
    new_dict = {"source": []}
    for cell in analyzed_dict["source"]:
        if (
            (
                "q1." in " ".join(cell["content_old"])
                and not "v18q1" in " ".join(cell["content_old"])
            )
            or "q2." in " ".join(cell["content_old"])
            or "q3" in " ".join(cell["content_old"])
            or "q4" in " ".join(cell["content_old"])
            or "q5" in " ".join(cell["content_old"])
            or "q6" in " ".join(cell["content_old"])
            or "q7" in " ".join(cell["content_old"])
            or "q8" in " ".join(cell["content_old"])
            or "q9" in " ".join(cell["content_old"])
            or "step_1" in " ".join(cell["content_old"])
            or "step_2" in " ".join(cell["content_old"])
            or "step_3" in " ".join(cell["content_old"])
            or "step_4" in " ".join(cell["content_old"])
            or "q_1" in " ".join(cell["content_old"])
            or "q_2" in " ".join(cell["content_old"])
            or "q_3" in " ".join(cell["content_old"])
            or "q_4" in " ".join(cell["content_old"])
            or "q_5" in " ".join(cell["content_old"])
            or "q_6" in " ".join(cell["content_old"])
            or "q_7" in " ".join(cell["content_old"])
            or "q_8" in " ".join(cell["content_old"])
            or "q_9" in " ".join(cell["content_old"])
            # A spanish cell
            or "bisseçao" in " ".join(cell["content_old"])
        ):
            continue
        else:
            new_dict["source"].append(cell)
    return new_dict


def load_pre_trained_models_dev():
    for tag in ALL_TAGS.keys():
        debug_print("Loading " + tag + " ...")
        classifier = xgb.XGBClassifier()
        classifier.load_model(
            "./src/pipeline_analyzer/jn_analyzer/resources/models/model_"
            + tag.replace("-", "_").replace(" ", "_")
            + "_boost-"
            + str(datetime.now().strftime("%Y-%m-%d"))
            + ".json"
        )
        CLASSIFIERS[tag] = classifier
        VECTORIZERS[tag] = joblib.load(
            "./src/pipeline_analyzer/jn_analyzer/resources/models/vectorizer_"
            + tag.replace("-", "_").replace(" ", "_")
            + "_boost-"
            + str(datetime.now().strftime("%Y-%m-%d"))
            + ".joblib"
        )


def load_pre_trained_models():
    for tag in ALL_TAGS.keys():
        debug_print("Loading " + tag + " ...")
        classifier = xgb.XGBClassifier()
        classifier.load_model(
            "./src/pipeline_analyzer/jn_analyzer/resources/models/model_"
            + tag.replace("-", "_").replace(" ", "_")
            + "_boost.json"
        )
        CLASSIFIERS[tag] = classifier
        VECTORIZERS[tag] = joblib.load(
            "./src/pipeline_analyzer/jn_analyzer/resources/models/vectorizer_"
            + tag.replace("-", "_").replace(" ", "_")
            + "_boost.joblib"
        )
