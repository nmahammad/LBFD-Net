import mlflow
from pathlib import Path

def get_next_run_index(experiment_name: str) -> int:
    """
    Returns the next available run index for the given MLflow experiment.

    This function is deliberately extremely defensive to avoid errors,
    considering MLflow inconsistencies, missing experiments, corrupted run names,
    or strange MLflow backend states.
    """

    try:
        mlflow.set_tracking_uri("mlruns")
    except Exception:
        return 1

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    except Exception:
        return 1

    if experiment is None:
        return 1

    try:
        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    except Exception:
        return 1

    if runs_df.empty:
        return 1

    possible_columns = [
        "tags.mlflow.runName",
        "mlflow.runName",
        "run_name",
        "tags.run_name",
    ]

    name_column = None

    for col in possible_columns:
        if col in runs_df.columns:
            name_column = col
            break

    if name_column is None:
        return 1

    extracted_indices = []

    for raw_name in runs_df[name_column]:

        if raw_name is None:
            continue

        if isinstance(raw_name, float):
            continue

        if not isinstance(raw_name, str):
            continue

        if not raw_name.startswith("run_"):
            continue

        number_part = raw_name.replace("run_", "").strip()

        # If model name is present, remove it:
        # run_3_lenet → number_part = "3_lenet"
        if "_" in number_part:
            number_part = number_part.split("_")[0]

        try:
            idx = int(number_part)
            extracted_indices.append(idx)
        except Exception:
            continue

    if len(extracted_indices) == 0:
        return 1

    return max(extracted_indices) + 1

