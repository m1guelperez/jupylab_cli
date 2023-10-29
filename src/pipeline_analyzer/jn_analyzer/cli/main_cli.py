import click
import warnings
import os

warnings.filterwarnings("ignore")
from pipeline_analyzer.jn_analyzer.hybrid_workflow import (
    benchmark_hybrid,
    start_pipeline_hybrid,
)
from pipeline_analyzer.jn_analyzer.constants import (
    PATH_TO_JSON_RESOURCE_INPUT_DIR,
    PATH_TO_NEW_TRAINED_MODELS,
)

from pipeline_analyzer.jn_analyzer.evaluation import (
    new_preprocessing_training_and_validation_data,
    big_evaluation,
)
from pipeline_analyzer.jn_analyzer.machine_learning_model import run_model
from pipeline_analyzer.jn_analyzer.utils import load_pre_trained_models


# Used to create a group, such that we can use @cli.command() to create subcommands
@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--path",
    default=PATH_TO_JSON_RESOURCE_INPUT_DIR,
    help="Custom path.",
)
@click.option(
    "--debug",
    default=False,
    help="Debugmode.",
)
@click.option(
    "--headers",
    default=True,
    help="Creating headers or tags.",
)
def label_notebooks(path, debug, headers):
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path.replace("inputs", "outputs")):
        os.mkdir(path.replace("inputs", "outputs"))
    print("Starting Hybrid solution.")
    load_pre_trained_models()
    start_pipeline_hybrid(path, debug, headers)


@cli.command()
def help():
    print(
        "Welcome to the CLI for the pipeline analyzer. The following commands are available:"
        "\n\nlabel_notebooks: This command will start the pipeline analyzer and will label all notebooks in the input folder."
        "\n\nbench: This command will start the benchmarking of the pipeline analyzer."
        "\n\neval: This command will start the evaluation of the pipeline analyzer."
    )


@cli.command()
@click.option(
    "--debug",
    default=False,
    help="Run in debug mode.",
)
@click.option(
    "--headers",
    default=True,
    help="Insert tags or headers.",
)
@click.argument("dataset", default="jupylab")
def bench(debug, headers, dataset):
    print("Starting benchmarking JupyLab.")
    benchmark_hybrid(debug, headers, dataset)


@cli.command()
def eval():
    print("Starting evaluation.")
    big_evaluation()


@cli.command()
@click.option(
    "--all",
    default="no",
    help="Run with cv script",
)
def new(all):
    if not os.path.isdir(PATH_TO_NEW_TRAINED_MODELS):
        os.mkdir(PATH_TO_NEW_TRAINED_MODELS)
    print(
        "Generating new training data from 120 notebooks and apply it to the 120 validation notebooks and the newly generated validation data from headergen."
    )
    new_preprocessing_training_and_validation_data()
    if all == "yes":
        print("Run model script.")
        run_model()
