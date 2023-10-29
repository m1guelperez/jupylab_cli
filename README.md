# JupyLabel Outline

## Research 
The corresponding paper can be found under: https://arxiv.org/abs/2403.07562

## Running the JupyLabel CLI using Conda
Setup the environment and install the project with the following commands:
```sh
conda create -n cli python=3.10
pip install .
pip install -r requirements.txt
```
Create the `inputs`, `outputs` and `backups` folders in `"./src/pipeline_analyzer/jn_analyzer/resources/"`.
```sh
mkdir inputs outputs backups
```

### Commands
Place all `.ipynb` files into the `inputs` folder. The default path is `"./src/pipeline_analyzer/jn_analyzer/resources/inputs/"`. 

All labeled Notebooks and intermediate files, when running in `DEBUG_MODE`, can be found in `"./src/pipeline_analyzer/jn_analyzer/resources/outputs/"` and `"./src/pipeline_analyzer/jn_analyzer/resources/backups/"`.

After that you can analyze the notebooks with the following command:
`analyze label-notebooks` to label all notebooks in the inputs folder. You can provide the following options:
```sh
--path # input folder path (default: "./src/pipeline_analyzer/jn_analyzer/resources/inputs/")
--debug # debug mode (default: False)
--headers # inserting headers instead of tags (default: True)
```

`analyze eval` to evaluate performance metrics.
`analyze bench {dataset}` to benchmark on 1000 notebooks. Dataset is either `jupylab` or `headergen`. When using `jupylab` it will benchmark on 1000 Jupyter Notebooks and when using `headergen` it will use the 15 notebooks provided by `headergen`.
This command also has the following options:
```sh
--debug # debug mode (default: False)
--headers # inserting headers instead of tags (default: True)
```
`analyze new` to prepare data from scratch to train the models. This is especially usefull, if you want to change the pre-processing of JupyLab and investigate how it influences the model performance. This command also provides the following option:
```sh
--all # Trains and evaluates the models after training them on the newly pre-processed data. The models are saved in the resources folder under new_trained_models (default: no)
```

When creating new models from scratch, simply delete/overwrite the models that currently exist in the /resources/models folder.

## Docker Installation
```sh
docker build -t jupylab .
docker run -it -v ${PWD}/jupylab:/jupylab jupylab bash
cd src/pipeline_analyzer/jn_analyzer/resources
mkdir inputs outputs backups
```