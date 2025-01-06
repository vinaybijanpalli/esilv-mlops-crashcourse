# %% [markdown]
# <h1>Part 4 - Experiment Tracking</h1>

# %% [markdown]
# # Experiment Tracking and Model Management with MLFlow

# %% [markdown]
# There are many ways to use the MLFlow Tracking API. For simple local uses, the best is to leave the data management to MLFlow and let it store runs, metrics, models and artifacts locally. For more advanced usage, all of this information can be stored in databases. You can find the detailed on MLFlow's documentation [here](https://mlflow.org/docs/latest/tracking.html#scenario-1-mlflow-on-localhost).

# %% [markdown]
# ## Exploring MLFlow
#
# MLflow setup:
# * Tracking server: no
# * Backend store: local filesystem
# * Artifacts store: local filesystem
#
# The experiments can be explored locally by launching the MLflow UI.

# %% [markdown]
# Let's print the tracking server URI, where the experiments and runs are going to be logged. We observe it refers to a local path.

# %%
import pickle
from typing import Any
import os
import gdown
from scipy.sparse import csr_matrix
from typing import List
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
import mlflow

print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

# %% [markdown]
# After this initialization, we can connect create a client to connect to the API and see what experiments are present.

# %% [markdown]
# By refering to mlflow's [documentation](https://mlflow.org/docs/latest/python_api/mlflow.client.html), create a client and display a list of the available experiments using the search_experiments function. This function could prove useful later to programatically explore experiments (rather than in the UI)

# %%


# Create an MLflow client
client = MlflowClient()

# Search for experiments
experiments = client.search_experiments()
experiments
# Display the list of available experiments
# print("Available Experiments:")
# for experiment in experiments:
#     print(f"- Name: {experiment.name}")
#     print(f"  ID: {experiment.experiment_id}")
#     print(f"  Artifact Location: {experiment.artifact_location}")
#     print(f"  Lifecycle Stage: {experiment.lifecycle_stage}")
#     print(f"  Creation Time: {experiment.creation_time}")
#     print(f"  Last Update Time: {experiment.last_update_time}")
#     print("---")

# %% [markdown]
# We see that there is a default experiment for which the runs are stored locally in the mlruns folder.

# %% [markdown]
# ### Creating an experiment and logging a new run

# %% [markdown]
# An experiment is a logical entity regrouping the logs of multiple attempts at solving a same problem, called runs. \
# We will now work with the classic sklearn dataset iris. Our goal here is to manage to classify the different iris species. To track our models performance, we will log every attempt as a "run" and create a new experiment "iris-experiment-1" to regroup them.

# %% [markdown]
# Lookup the mlflow.run and mlflow.start_run functions [here](https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=start_run#mlflow.start_run) to find out how to manage runs.
# Explore [this part](https://mlflow.org/docs/latest/python_api/mlflow.html) to learn more about the log_params, log_metrics and log_artifact functions. Find out how to log sklearn models [here](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html])

# %% [markdown]
# Complete the following in order to log the parameters, interesting metrics and the model.

# %%

mlflow.set_experiment("iris-experiment-1")

with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set and log parameters
    params = {"C": 0.1, "random_state": 42}
    mlflow.log_params(params)

    # Train the model
    model = LogisticRegression(**params).fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # print(f"Run ID: {run_id}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")

# %%
experiments = client.search_experiments()
experiments

# %% [markdown]
# Try running the training script with various parameters to have runs to compare.
# You can now explore your run(s) using the ui: \
# (Paste "mlflow ui --host 0.0.0.0 --port 5002" in your terminal, or run the cell below)
#
# **N.B.** Make sure you are in the lecture folder and not the repo root!

# %%
!mlflow ui - -host 0.0.0.0 - -port 5002

# %% [markdown]
# You will have to kill the cell to continue experimenting

# %% [markdown]
# ### Interacting with the model registry

# %% [markdown]
# If you are satisfied with the last run's model, you can transform the logged model into a registered model. It will be logged in the Model Registry, which makes it easier to use in production and manage versions.

# %%
# We already have our run id from above. Let's use it to register the model

result = mlflow.register_model(f"runs:/{run_id}/models", "iris_lr_model")

# %% [markdown]
# # Use Case

# %% [markdown]
# Now we will get back to our taxi rides use case:

# %%


# %% [markdown]
# ## 0 - Download Data

# %%
!pip install gdown

# %%

DATA_FOLDER = "../../data"
train_path = f"{DATA_FOLDER}/yellow_tripdata_2021-01.parquet"
test_path = f"{DATA_FOLDER}/yellow_tripdata_2021-02.parquet"
predict_path = f"{DATA_FOLDER}/yellow_tripdata_2021-03.parquet"


if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"New directory {DATA_FOLDER} created!")

gdown.download(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet",
    train_path,
    quiet=False,
)
gdown.download(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-02.parquet",
    test_path,
    quiet=False,
)
gdown.download(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-03.parquet",
    predict_path,
    quiet=False,
)

# %% [markdown]
# ## 1 - Load data

# %%


def load_data(path: str):
    return pd.read_parquet(path)


train_df = load_data(train_path)
train_df.head()

# %% [markdown]
# ## 2 - Prepare the data

# %% [markdown]
# Let's prepare the data to make it Machine Learning ready. \
# For this, we need to clean it, compute the target (what we want to predict), and compute some features to help the model understand the data better.

# %% [markdown]
# ### 2-1 Compute the target

# %% [markdown]
# We want to predict a taxi trip duration in minutes. Let's compute it as a difference between the drop-off time and the pick-up time for each trip.

# %%


def compute_target(
    df: pd.DataFrame,
    pickup_column: str = "tpep_pickup_datetime",
    dropoff_column: str = "tpep_dropoff_datetime",
) -> pd.DataFrame:
    df["duration"] = df[dropoff_column] - df[pickup_column]
    df["duration"] = df["duration"].dt.total_seconds() / 60
    return df


train_df = compute_target(train_df)

# %%
train_df["duration"].describe()

# %% [markdown]
# Let's remove outliers and reduce the scope to trips between 1 minute and 1 hour

# %%
MIN_DURATION = 1
MAX_DURATION = 60


def filter_outliers(df: pd.DataFrame, min_duration: int = 1, max_duration: int = 60) -> pd.DataFrame:
    return df[df["duration"].between(min_duration, max_duration)]


train_df = filter_outliers(train_df)

# %% [markdown]
# ### 2-2 Prepare features

# %% [markdown]
# #### 2-2-1 Categorical features

# %% [markdown]
# Most machine learning models don't work with categorical features. Because of this, they must be transformed so that the ML model can consume them.

# %%
CATEGORICAL_COLS = ["PUlocationID", "DOlocationID"]


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["PULocationID", "DOLocationID", "passenger_count"]
    df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


train_df = encode_categorical_cols(train_df)

# %%


def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> dict:

    if categorical_cols is None:
        categorical_cols = ["PULocationID", "DOLocationID", "passenger_count"]
    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["duration"].values

    x = dv.transform(dicts)
    return x, y, dv


X_train, y_train, dv = extract_x_y(train_df)

# %% [markdown]
# ## 3 - Train model

# %% [markdown]
# We train a basic linear regression model to have a baseline performance

# %%


def train_model(x_train: csr_matrix, y_train: np.ndarray):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr


model = train_model(X_train, y_train)

# %% [markdown]
# ## 4 - Evaluate model

# %% [markdown]
# We evaluate the model on train and test data

# %% [markdown]
# ### 4-1 On train data

# %%


def predict_duration(input_data: csr_matrix, model: LinearRegression):
    return model.predict(input_data)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    return mean_squared_error(y_true, y_pred, squared=False)


prediction = predict_duration(X_train, model)
train_me = evaluate_model(y_train, prediction)
train_me

# %% [markdown]
# ### 4-2 On test data

# %%
test_df = load_data(test_path)

# %%
test_df = compute_target(test_df)
test_df = encode_categorical_cols(test_df)
X_test, y_test, _ = extract_x_y(test_df, dv=dv)

# %%
y_pred_test = predict_duration(X_test, model)
test_me = evaluate_model(y_test, y_pred_test)
test_me

# %% [markdown]
# ## 5 - Log Model Parameters to MlFlow

# %% [markdown]
# Now that all our development functions are built and tested, let's create a training pipeline and log the training parameters, logs and model to MlFlow.

# %% [markdown]
# Create a training flow, log all the important parameters, metrics and model. Try to find what could be important and needs to be logged.

# %%

# Set the experiment name
mlflow.set_experiment("taxi-duration-prediction")

# Start a run
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Set tags for the run
    mlflow.set_tag("developer", "YourName")
    mlflow.set_tag("dataset", "NYC Yellow Taxi 2021")

    # Load data
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    # Log data info
    mlflow.log_param("train_data_path", train_path)
    mlflow.log_param("test_data_path", test_path)
    mlflow.log_param("train_data_rows", len(train_df))
    mlflow.log_param("test_data_rows", len(test_df))

    # Compute target
    train_df = compute_target(train_df)
    test_df = compute_target(test_df)

    # Filter outliers
    MIN_DURATION = 1
    MAX_DURATION = 60
    train_df = filter_outliers(train_df, MIN_DURATION, MAX_DURATION)
    test_df = filter_outliers(test_df, MIN_DURATION, MAX_DURATION)

    mlflow.log_param("min_duration", MIN_DURATION)
    mlflow.log_param("max_duration", MAX_DURATION)
    mlflow.log_param("filtered_train_data_rows", len(train_df))
    mlflow.log_param("filtered_test_data_rows", len(test_df))

    # Encode categorical columns
    # Check for the correct column names
    possible_pu_columns = ["PULocationID", "PUlocationID", "pulocationid"]
    possible_do_columns = ["DOLocationID", "DOlocationID", "dolocationid"]

    pu_column = next((col for col in possible_pu_columns if col in train_df.columns), None)
    do_column = next((col for col in possible_do_columns if col in train_df.columns), None)

    if pu_column is None or do_column is None:
        raise ValueError(f"Required location columns not found. Available columns: {train_df.columns}")

    CATEGORICAL_COLS = [pu_column, do_column]

    def encode_categorical_cols(df, categorical_cols):
        df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
        df[categorical_cols] = df[categorical_cols].astype("str")
        return df

    train_df = encode_categorical_cols(train_df, CATEGORICAL_COLS)
    test_df = encode_categorical_cols(test_df, CATEGORICAL_COLS)

    mlflow.log_param("categorical_columns", CATEGORICAL_COLS)

    # Extract X and y
    X_train, y_train, dv = extract_x_y(train_df, CATEGORICAL_COLS)
    X_test, y_test, _ = extract_x_y(test_df, CATEGORICAL_COLS, dv)

    # Train model
    model = train_model(X_train, y_train)

    # Log model parameters
    mlflow.log_params(model.get_params())

    # Evaluate model on train set
    y_pred_train = predict_duration(X_train, model)
    train_rmse = evaluate_model(y_train, y_pred_train)
    mlflow.log_metric("train_rmse", train_rmse)

    # Evaluate model on test set
    y_pred_test = predict_duration(X_test, model)
    test_rmse = evaluate_model(y_test, y_pred_test)
    mlflow.log_metric("test_rmse", test_rmse)

    # Log your model
    mlflow.sklearn.log_model(model, "taxi_duration_model")

    # Log feature names
    # Use get_feature_names_out() for newer scikit-learn versions
    # try:
    #     feature_names = dv.get_feature_names_out()
    # except AttributeError:
    #     # Fallback for older versions
    #     feature_names = dv.get_feature_names()

    # mlflow.log_param("feature_names", feature_names.tolist())

    print(f"Run ID: {run_id}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Register your model in MLflow model registry
    model_uri = f"runs:/{run_id}/taxi_duration_model"
    registered_model = mlflow.register_model(model_uri, "taxi_duration_predictor")
    print(f"Model registered as: {registered_model.name}")

# %% [markdown]
# If the model is satisfactory, we stage it as production using the appropriate version. This will help us retreiving it for predictions.

# %% [markdown]
# Create a mlflow client and use the [mlflow documentation](https://mlflow.org/docs/latest/python_api/mlflow.client.html?highlight=transition_model_version_stage#mlflow.client.MlflowClient.transition_model_version_stage) to stage the appropriate model as being in "production".

# %%

# Create an MLflow client
client = MlflowClient()

# Set the model name
model_name = "taxi_duration_predictor"

# Get all versions of the model
model_versions = client.search_model_versions(f"name='{model_name}'")

if not model_versions:
    print(f"No versions found for model '{model_name}'")
else:
    # Sort versions by creation timestamp (newest first)
    sorted_versions = sorted(model_versions, key=lambda x: x.creation_timestamp, reverse=True)

    # Get the latest version
    latest_version = sorted_versions[0]

    # Transition the latest version to "Production" stage
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production"
    )

    print(f"Model '{model_name}' version {latest_version.version} transitioned to Production stage")

    # Optional: Archive other versions
    for version in sorted_versions[1:]:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )
        print(f"Model '{model_name}' version {version.version} archived")

# Verify the current production model
production_model = client.get_latest_versions(model_name, stages=["Production"])
if production_model:
    print(f"Current production model: {model_name} version {production_model[0].version}")
else:
    print("No production model found")

# %% [markdown]
# ## 6 - Predict

# %% [markdown]
# We can now use our model to predict on fresh unseen data and forecast what is going to be the duration of a tawi trip depending on trip characteristics.

# %%
# Load prediction data
predict_df = load_data(predict_path)

# Apply feature engineering
CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]  # Make sure these column names match your dataset
predict_df = encode_categorical_cols(predict_df, CATEGORICAL_COLS)
X_pred, _, _ = extract_x_y(predict_df, categorical_cols=CATEGORICAL_COLS, dv=dv, with_target=False)

# Load production model
model_name = "taxi_duration_predictor"
production_model = mlflow.sklearn.load_model(f"models:/{model_name}/production")

# Make predictions
y_pred = predict_duration(X_pred, production_model)
print(y_pred[:10])  # Print first 10 predictions

# %% [markdown]
# ## 7 - To go further

# %% [markdown]
# If you managed to go this far, you can try solving the use case using an other regression model like [XGBoost](https://xgboost.readthedocs.io/en/stable/) for instance.

# %% [markdown]
#

# %%
############################################


def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


save_pickle("dv__v0.0.5.pkl", dv)
save_pickle("model__v0.0.5.pkl", model)

############################################
