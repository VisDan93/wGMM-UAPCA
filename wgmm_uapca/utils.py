import numpy as np
import pandas as pd
import json
import shutil
import requests
from pathlib import Path
from sklearn.mixture import GaussianMixture

def load_dataset(name: str, json_file: str = "data/gmm_components.json") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a single dataset from the mespadoto repository.
    Downloads it if not found locally.

    Parameters
    ----------
    name : str
        Dataset name to load.
    json_file : str
        Path to the gmm_components.json file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Label vector.
    """
    # Load available dataset list
    with open(json_file, "r") as f:
        gmm_config = json.load(f)
    
    if name not in gmm_config:
        raise ValueError(f'Dataset "{name}" not available in mespadoto.')
    
    if name == "hatespeech_demo":
        name = "hatespeech"

    dataset_path = Path("data") / "datasets" / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    x_file = dataset_path / "X.csv.gz"
    y_file = dataset_path / "y.csv.gz"

    # Download missing files
    for file_path, fname in [(x_file, "X.csv.gz"), (y_file, "y.csv.gz")]:
        if not file_path.exists():
            url = f"https://mespadoto.github.io/proj-quant-eval/data/{name}/{fname}"
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download {url}")
            with open(file_path, "wb") as f_out:
                shutil.copyfileobj(response.raw, f_out)

    # Load CSVs
    X = pd.read_csv(x_file, compression="gzip")
    y = pd.read_csv(y_file, compression="gzip")

    X.columns = [f"Feature {i+1}" for i in range(X.shape[1])]
    y.columns = ["Label"]
    y["Label"] = y["Label"].astype(int)

    return X, y


def load_datasets(names: list[str] | None = None, json_file: str = "data/gmm_components.json") -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load multiple datasets using load_dataset().

    Parameters
    ----------
    names : list[str] or None
        List of dataset names to load. If None, all datasets in the json file are loaded.
    json_file : str
        Path to the gmm_components.json file.

    Returns
    -------
    datasets : dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Mapping dataset_name -> (X, y).
    """
    # Load dataset names from json
    with open(json_file, "r") as f:
        gmm_config = json.load(f)

    # Decide which datasets to load
    all_datasets = list(gmm_config.keys())
    if names is None:
        datasets_to_load = all_datasets
    else:
        # Check if requested datasets exist in json
        missing = [n for n in names if n not in all_datasets]
        if missing:
            raise ValueError(f'Dataset(s) "{missing}" not available in mespadoto.')
        datasets_to_load = names

    datasets = {name: load_dataset(name, json_file=json_file) for name in datasets_to_load}

    return datasets


def fit_gmms(dataset_name: str, data: tuple[pd.DataFrame, pd.DataFrame], json_file: str = "data/gmm_components.json") -> dict[int, GaussianMixture]:
    """
    Fit Gaussian Mixture Models (GMMs) for a single dataset and each label
    according to the specifications in gmm_components.json.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    data : tuple[pd.DataFrame, pd.DataFrame]
        Tuple (X, y) with pandas DataFrames.
    json_file : str
        Path to the gmm_components.json file.

    Returns
    -------
    gmms : dict[int, GaussianMixture]
        Mapping label -> fitted GMM for this dataset.
    """
    X, y = data
    y = y.squeeze()

    # Load configuration
    with open(json_file, "r") as f:
        gmm_config = json.load(f)

    if dataset_name not in gmm_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in {json_file}.")

    gmms = {}
    label_config = gmm_config[dataset_name]

    for label_str, params in label_config.items():
        label = int(label_str)
        n_components = params["n_components"]
        random_state = params["random_state"]

        # Select samples for this label
        X_label = X[y == label]
        if X_label.shape[0] == 0:
            raise ValueError(f"No samples found for dataset '{dataset_name}', label {label}")

        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-5,
        )
        gmm.fit(X_label)
        gmms[label] = gmm

    return gmms


def calculate_grid(distributions: dict[str, GaussianMixture], P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate dynamic bounds for visualization based on projected GMM samples.

    Parameters
    ----------
    distributions : dict[str, GaussianMixture]
        Mapping label -> fitted GMM.
    P : np.ndarray
        Projection matrix of shape (d, 2).

    Returns
    -------
    x, y : np.ndarray
        Coordinate arrays for the 2D grid.
    """
    padding = 0.2
    n_samples = 1000
    grid_size = 150

    # Collect projected samples
    projected_points = []
    for gmm in distributions.values():
        samples, _ = gmm.sample(n_samples)
        projected_points.append(samples @ P)
    projected_points = np.vstack(projected_points)

    # Compute min/max and padding
    x_min, x_max = projected_points[:, 0].min(), projected_points[:, 0].max()
    y_min, y_max = projected_points[:, 1].min(), projected_points[:, 1].max()
    x_pad = padding * (x_max - x_min)
    y_pad = padding * (y_max - y_min)

    # Create grid
    x = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    y = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)

    return x, y