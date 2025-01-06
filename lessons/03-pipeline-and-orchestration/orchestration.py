# %% [markdown]
# # Machine Learning Pipelines and Orchestration with Prefect

# %% [markdown]
# > ⚠️  **Version**: This module has been created using Prefect 2.13.7

# %% [markdown]
# ## 0 - Useful functions

# %% [markdown]
# ### 0.1 - From previous lessons
#

# %%
# lib/config.py
from prefect import flow, serve
from prefect import flow, task
import httpx
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from typing import Optional
import pickle
from typing import Any
from sklearn.feature_extraction import DictVectorizer
from loguru import logger
import scipy.sparse
import pandas as pd
import numpy as np
import os
from typing import List, Tuple
CATEGORICAL_COLS = ["PULocationID", "DOLocationID", "passenger_count"]

DATA_DIRPATH = "../../data"
MODELS_DIRPATH = "../../models"

# %%
# lib/preprocessing.py


def compute_target(
    df: pd.DataFrame, pickup_column: str = "tpep_pickup_datetime", dropoff_column: str = "tpep_dropoff_datetime"
) -> pd.DataFrame:
    """Compute the trip duration in minutes based on pickup and dropoff time"""
    df["duration"] = df[dropoff_column] - df[pickup_column]
    df["duration"] = df["duration"].dt.total_seconds() / 60
    return df


def filter_outliers(df: pd.DataFrame, min_duration: int = 1, max_duration: int = 60) -> pd.DataFrame:
    """
    Remove rows corresponding to negative/zero
    and too high target' values from the dataset
    """
    return df[df["duration"].between(min_duration, max_duration)]


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """Encode categorical columns as strings"""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df.loc[:, categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df.loc[:, categorical_cols] = df[categorical_cols].astype("str")
    return df


def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, DictVectorizer]:
    """Extract X and y from the dataframe"""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["duration"].values

    x = dv.transform(dicts)
    return x, y, dv


def process_data(filepath: str, dv=None, with_target: bool = True) -> scipy.sparse.csr_matrix:
    """
    Load data from a parquet file
    Compute target (duration column) and apply threshold filters (optional)
    Turn features to sparce matrix
    :return The sparce matrix, the target' values and the
    dictvectorizer object if needed.
    """
    df = pd.read_parquet(filepath)
    if with_target:
        logger.debug(f"{filepath} | Computing target...")
        df1 = compute_target(df)
        logger.debug(f"{filepath} | Filtering outliers...")
        df2 = filter_outliers(df1)
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df3 = encode_categorical_cols(df2)
        logger.debug(f"{filepath} | Extracting X and y...")
        return extract_x_y(df3, dv=dv)
    else:
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df1 = encode_categorical_cols(df)
        logger.debug(f"{filepath} | Extracting X and y...")
        return extract_x_y(df1, dv=dv, with_target=with_target)

# %% [markdown]
# ### 0-2 Helpers for this session

# %% [markdown]
# You also have other helpers to show you prefect's features in the `helpers.py` file.


# %%
# lib/helpers.py


def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# %% [markdown]
# ## 1 - Create workflow functions

# %% [markdown]
# Create five functions to complete the ML process :
# - `train_model`
# - `predict`
# - `evaluate_model`
# - A workflow function to perform the whole training process `train_model_workflow`
#     - Process data
#     - Train model
#     - Evaluate model
# - A workflow function to perform the whole prediction process `batch_predict_workflow`
#     - Process data without target column
#     - Predict
#
#
# For the last two functions, you can start without saving / loading artifacts add these steps after.
# Please think about what artifacts you'll need to save and load to pass from training to predict workflows.
#
# Start by coding these functions here in the notebook
#
# Then, test your code with the downloaded data (e.g. January to train and February to predict).
#
# Finally, copy your code in the `lib` folder in the `modeling.py` and `workflows.py` files and test your workflows again using such a command:
#
# ```bash
# python lib/workflows.py
# ```


# %%


def train_model(X: scipy.sparse.csr_matrix, y: np.ndarray) -> LinearRegression:
    """..."""
    ...


def predict(X: scipy.sparse.csr_matrix, model: LinearRegression) -> np.ndarray:
    """..."""
    ...


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """..."""
    ...


def train_model_workflow(
    train_filepath: str,
    test_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    """..."""
    ...


def batch_predict_workflow(
    input_filepath: str,
    model: Optional[LinearRegression] = None,
    dv: Optional[DictVectorizer] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    """..."""
    ...

# %% [markdown]
# ## 2 - Setup and explore Prefect
#
# We are going to use [Prefect](https://docs.prefect.io/2.6/tutorials/first-steps/), an Open Source orchestration tool with a Python SDK.
#
#
# **WINDOWS USERS**:
#
# You might run into issues with Prefect on Windows. If you do, please follow [Prefects instructions](https://docs.prefect.io/2.13.7/getting-started/installation/#install-prefect) to install Prefect on your machine

# %% [markdown]
# ### 2-1 Setup Prefect UI

# %% [markdown]
# Before starting to implement tasks and flows with prefect, let's set up the UI in order to have a good visualization of our work.
#
# Steps :
#
# - Set an API URL for your local server to make sure that your workflow will be tracked by this specific instance :
# ```
# prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
# ```
#
# - Check you have SQLite installed ([Prefect backend database system](https://docs.prefect.io/2.13.7/getting-started/installation/#external-requirements)):
# ```
# sqlite3 --version
# ```
#
# - Start a local prefect server :
# ```
# prefect server start --host 0.0.0.0
# ```
#
# If you want to reset the database, run :
# ```
# prefect server database reset
# ```
#
#
# You can visit the UI at http://0.0.0.0:4200/dashboard
#
# ![](images/starting_page.png)

# %% [markdown]
# ### 2-2 Prefect tasks and flows

# %% [markdown]
# [Prefect uses tasks and flows to build workflows](https://docs.prefect.io/2.13.7/tutorial/flows/).
# - Flows are like functions. They can take inputs, perform work, and return an output. In fact, you can turn any function into a Prefect flow by adding the @flow decorator
# - A task is any Python function decorated with a @task decorator called within a flow. You can think of a flow as a recipe for connecting a known sequence of tasks together. Tasks, and the dependencies between them, are displayed in the flow run graph, enabling you to break down a complex flow into something you can observe, understand and control at a more granular level.
#     - All tasks must be called from within a flow. Tasks may not call other tasks directly.
#     - Not all functions in a flow need be tasks. Use them only when their features are useful.


# %%
# Example


@task
def get_url(url: str, params: dict = None):
    response = httpx.get(url, params=params)
    response.raise_for_status()
    return response.json()


@flow(retries=3, retry_delay_seconds=5, log_prints=True)
def get_repo_info(repo_name: str = "PrefectHQ/prefect"):
    url = f"https://api.github.com/repos/{repo_name}"
    repo_stats = get_url(url)
    print(f"{repo_name} repository statistics 🤓:")
    print(f"Stars 🌠 : {repo_stats['stargazers_count']}")
    print(f"Forks 🍴 : {repo_stats['forks_count']}")

# %% [markdown]
# ## 3 - Create Prefect tasks and flows

# %% [markdown]
# ### 3-1 Create tasks and flows

# %% [markdown]
# Use the decorators `@task` and `@flow` to create your first prefect flow : The Processing flow.
#
# Prefect will try to use by default different thread to run each task. If you want sequential steps, introduce this dependencies through the name of each task output.
#
#
# Steps:
# - Create a task for each function you created in the previous section. You can start by doing these in the notebook.
# - Test your code by calling the flows run with downloaded data (this can be done in the notebook too).
# - Update your files in the `lib` folder. You should now have completed all files except `deployment.py`.
#
#
# You can see registered flows in the UI :
# ![Flows in Prefect UI](images/flows_ui.png)
#
#
# And visualize the run of a flow :
# ![Flows in Prefect UI](images/flow_run.png)
#
#
# > [!Warning]
# > **Typing tasks and flows in prefect** :
# > Typing tasks in prefect is done as with any python code.
# > For flows, either use `validate_parameters=False` or define pydantic models for prefect to understand your NON DEFAULT typing (see extra section).
# > But if all tasks are typed, since flows are just set of tasks, it should be all good if we don't want to add a layer of complexity
# > `Default types` : str, int ...
#
#
#

# %% [markdown]
# ### 3-2 Customize your flows

# %% [markdown]
# You can configure the properties and special behavior for your prefect tasks/flow in the decorator.
# For example, you can tell if you want to retry on a failure, set name or tags, etc... \
# An example is given in the `helpers.py` file.
# ```
# @task('name=failure_task', tags=['fails'], retries=3, retry_delay_seconds=60)
# def func():
#   ...
#
# ```
#
# - Add names, tasks, and desired behavior to your tasks/flows
# - Test your code
# - Visualize in the local prefect UI
#
# If a task fails in the flow, it is possible to visualize which task fail and access the full log and traceback error
# by clicking on the tasks. \
# We can also access run information inside de `state` object that can be returned by the flows using python code

# %% [markdown]
# ## 4 - Deploy your flows

# %% [markdown]
# Now that all the workflows are defined, we can now schedule automatics runs for these pipelines.
# Let's assume that we have a process that tells us that our model need to be retrained weekly based on some performance analysis. We also receive data to predict each hour.
#
# Use prefect deployment object in order to :
# - Schedule complete ml process to run weekly
# - Schedule prediction pipeline to run each hour
#
#
# **Please note that you can test your code with the `to_deployment` here, however you'll have to move to scripts to test the deployment with `serve`.**
#
# You can deploy your flows by following [Prefect documentation here](https://docs.prefect.io/2.13.7/tutorial/deployments/#running-multiple-deployments-at-once).
#
# ⚠️  Serving a model with prefect is a long-running command, meaning that you will need to run it in a separate terminal or in the background.
# Interupting the command will stop the deployment, but you'll be still be able to see it the UI.
#
# In the UI, you should be able see deployments:
# ![Deployments in Prefect UI](images/deployments.png)
#
#
# And the scheduled runs for one deployment:
# ![Scheduled runs in Prefect UI](images/scheduled_runs.png)


# %%
# hello_world.py


@flow(name="Hello world")
def hello_world(name: str = "world"):
    print(f"Hello {name}!")


if __name__ == "__main__":
    hello_world_deployment = hello_world.to_deployment(
        name='Hello world Deployment',
        version='0.1.0',
        tags=['hello world'],
        interval=600
        parameters={
            'name': 'John Doe'
        }
    )
    # Above: can be tested in notebook. Below: must be called from python script
    serve(hello_world_deployment)


# %% [markdown]
# ## 5 - Extra concepts

# %% [markdown]
# ### 5-1 Prefect workers

# %% [markdown]
# ### 5-2 Prefect typing using Pydantic
