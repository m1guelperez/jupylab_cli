# JupyLabel Outline

## Running the JupyLabel CLI using Conda
Setup the environment and install the project with the following commands:
```bash
conda create -n cli python=3.10
pip install .
pip install -r requirements.txt
```
Create the `inputs`, `outputs` and `backups` folders in `"./src/pipeline_analyzer/jn_analyzer/resources/"`.
```bash
mkdir inputs outputs backups
```

### Commands
Place all `.ipynb` files into the `inputs` folder. The default path is `"./src/pipeline_analyzer/jn_analyzer/resources/inputs/"`. 

All labeled Notebooks and intermediate files, when running in `DEBUG_MODE`, can be found in `"./src/pipeline_analyzer/jn_analyzer/resources/outputs/"` and `"./src/pipeline_analyzer/jn_analyzer/resources/backups/"`.

After that you can analyze the notebooks with the following command:
`analyze label-notebooks` to label all notebooks in the inputs folder. You can provide the following options:
```bash
--path # input folder path (default: "./src/pipeline_analyzer/jn_analyzer/resources/inputs/")
--debug # debug mode (default: False)
--headers # inserting headers instead of tags (default: True)
```

`analyze eval` to evaluate performance metrics.
`analyze bench <solution>` to benchmark on 1000 notebooks. The solution argument must be either `old` or `new`, where `new` is the implementation using classes but currently is slower.
This command also has the following options:
```bash
--debug # debug mode (default: False)
--headers # inserting headers instead of tags (default: True)
```

## Docker Installation
```bash
docker build -t notebook-labeling .
docker run -it -v ${PWD}/notebook-labeling:/notebook-labeling notebook-labeling bash
cd src/pipeline_analyzer/jn_analyzer/resources
mkdir inputs outputs backups
```