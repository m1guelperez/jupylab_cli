In this folder we the files that end with `concat.json` are the code concatenated code cells from the notebooks in `notebooks_headergen_json`.

Files that end with `concat_with_our_labels.json` are the concatenated files that contain:
* Raw content
* Processed content
* JupyLabels predicted tag
* JupyLabels source of truth tag
* HeaderGens predicted tag
* HeaderGens source of truth tag

The `evaluation_file.json` is a concatenation of all `concat_with_our_labels.json` files. This file is used for the evaluation, since here is everything bundled together.

The `source_of_truth_jupylabel.json` file contains all concatenated code cells of the notebooks and was manually corrected. Thus it is our source of truth. 