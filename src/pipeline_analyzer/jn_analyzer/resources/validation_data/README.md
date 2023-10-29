The file ending with `_SHORT` is used as source of truth. It only contains the original non-pre-processed content, tags and the output type. All tags in the `_SHORT` file have been manually corrected, this makes it easier to check the cells for correctness and edit the tags accordingly.

The file ending with `_COMPLETE` adds the pre-processed code cells. The rest of the fields are addopted by the `_SHORT` file. It is used for evaluation since, there is already the processed content present. Further, it allows us to investigate if the processing is working as intendet.

When making changes, edit the `_SHORT` file and execute `analyze new all --yes`. This command will then update the `_COMPLETE` file.

The folder `csvs` contains the csv files that are generated from the `_COMPLETE` file. The `error` folder presents all code cells, that were not correctly labeled by our model.