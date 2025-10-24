import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, gaussian_kde

def _calculate_overall_mean_cov_gmm(gmm: GaussianMixture) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the overall mean and covariance of a Gaussian Mixture Model.

    Parameters
    ----------
    gmm : GaussianMixture
        The fitted GMM.

    Returns
    -------
    mean : np.ndarray
        Overall mean vector of shape (d,)
    cov : np.ndarray
        Overall covariance matrix of shape (d, d)
    """
    overall_mean = np.sum(gmm.weights_[:, None] * gmm.means_, axis=0)

    weighted_cov = np.sum(gmm.weights_[:, None, None] * gmm.covariances_, axis=0)
    mean_diff = gmm.means_ - overall_mean
    weighted_outer = np.sum(gmm.weights_[:, None, None] * (mean_diff[:, :, None] @ mean_diff[:, None, :]), axis=0)

    overall_cov = weighted_cov + weighted_outer
    
    return overall_mean, overall_cov


def _calculate_ua_cov(means: np.ndarray, covs: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Compute the uncertainty-aware covariance matrix.

    Parameters
    ----------
    means : np.ndarray
        Array of shape (n, d) with the mean vectors.
    covs : np.ndarray
        Array of shape (n, d, d) with covariance matrices.
    weights : np.ndarray, optional
        Array of shape (n,) with weights for each label. Defaults to uniform.

    Returns
    -------
    ua_cov : np.ndarray
        Uncertainty-aware covariance matrix of shape (d, d)
    """
    n, d = means.shape
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / weights.sum()

    # Weighted global mean
    mu = np.sum(weights[:, None] * means, axis=0)

    # Centering matrix
    centering = np.outer(mu, mu)

    # Average within-label covariance
    avg_cov = np.sum(weights[:, None, None] * covs, axis=0)

    # Sample covariance of the means
    sample_cov = np.sum(weights[:, None, None] *
                        np.array([np.outer(means[i], means[i]) for i in range(n)]),
                        axis=0)

    # Uncertainty-aware covariance
    ua_cov = sample_cov + avg_cov - centering

    return ua_cov


def calculate_projmat(distributions: dict[str, GaussianMixture], weights: np.ndarray = None, n_dims: int = 2) -> np.ndarray:
    """
    Compute the projection matrix using the uncertainty-aware covariance.

    Parameters
    ----------
    distributions : dict
        Dictionary of label -> GaussianMixture.
    weights : np.ndarray, optional
        Label weights for computing UA covariance. Defaults to uniform.
    n_dims : int
        Number of target dimensions.

    Returns
    -------
    proj_mat : np.ndarray
        Projection matrix of shape (d, n_dims)
    """
    means = []
    covs = []

    for dist in distributions.values():
        mean, cov = _calculate_overall_mean_cov_gmm(dist)
        means.append(mean)
        covs.append(cov)

    means = np.array(means)
    covs = np.array(covs)

    if weights is None:
        weights = np.ones(means.shape[0]) / means.shape[0]

    ua_cov = _calculate_ua_cov(means, covs, weights)

    # SVD to get principal components
    eigvecs = np.linalg.svd(ua_cov, full_matrices=True)[0]

    return eigvecs[:, :n_dims]


def wgmm_uapca(distributions: dict[str, GaussianMixture], P: np.ndarray) -> dict[str, list[tuple[np.ndarray, np.ndarray, float]]]:
    """
    Perform Weighted GMM-based Uncertainty-Aware PCA.

    Each GMM is projected with UAPCA, retaining all components.

    Parameters
    ----------
    distributions : dict
        Dictionary of label -> GaussianMixture.
    P : np.ndarray
        Projection matrix of shape (d, n_dims).

    Returns
    -------
    projected_distributions : dict
        Dictionary mapping label -> list of (mean, cov, weight) tuples
        where mean is (n_dims,), cov is (n_dims, n_dims), weight is float.
    """
    # Project each GMM's components
    projected_distributions = {}
    for label, gmm in distributions.items():
        components = []
        for mu, cov, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            projected_mu = P.T @ mu
            projected_cov = P.T @ cov @ P
            components.append((projected_mu, projected_cov, w))
        projected_distributions[label] = components

    return projected_distributions


def uapca(distributions: dict[str, GaussianMixture], P: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Perform Uncertainty-Aware PCA on collapsed GMMs.

    Each GMM is reduced to a single Gaussian (overall mean + covariance),
    and then projected with UAPCA.

    Parameters
    ----------
    distributions : dict
        Dictionary of label -> GaussianMixture.
    P : np.ndarray
        Projection matrix of shape (d, n_dims).

    Returns
    -------
    projected_distributions : dict
        Dictionary mapping label -> (mean, cov) in projected space,
        where mean has shape (n_dims,), cov has shape (n_dims, n_dims).
    """
    # Compute overall means and covariances
    overall_means = []
    overall_covs = []
    for gmm in distributions.values():
        mean, cov = _calculate_overall_mean_cov_gmm(gmm)
        overall_means.append(mean)
        overall_covs.append(cov)
    
    overall_means = np.array(overall_means)
    overall_covs = np.array(overall_covs)

    # Project overall means and covariances
    projected_distributions = {}
    for label, (mean, cov) in zip(distributions.keys(), zip(overall_means, overall_covs)):
        projected_mean = P.T @ mean
        projected_cov = P.T @ cov @ P
        projected_distributions[label] = (projected_mean, projected_cov)

    return projected_distributions


def pca(data: tuple[pd.DataFrame, pd.DataFrame], P: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform standard PCA by projecting the raw dataset.

    Parameters
    ----------
    data : tuple[pd.DataFrame, pd.DataFrame]
        Tuple (X, y) with pandas DataFrames.
    P : np.ndarray
        Projection matrix of shape (d, n_dims).

    Returns
    -------
    (X_proj, y) : tuple[pd.DataFrame, pd.DataFrame]
        Projected data X_proj and original labels y.
    """
    # Project the raw data
    X, y = data
    X_proj = X.values @ P
    X_proj = pd.DataFrame(X_proj, index=X.index, columns=[f"PC{i+1}" for i in range(X_proj.shape[1])])

    return (X_proj, y)


def calculate_density(projection, grid: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Calculate density on a grid given a projection.

    Parameters
    ----------
    projection : dict or tuple
        - If dict[label] = list[(mu, cov, w)], interpret as WGMM-UAPCA.
        - If dict[label] = (mu, cov), interpret as UAPCA.
        - If tuple (X_proj, y), interpret as PCA raw projection.
    grid : tuple[np.ndarray, np.ndarray]
        (xx, yy) meshgrid arrays.

    Returns
    -------
    densities : dict[str, np.ndarray]
        Dictionary mapping label -> 2D density array.
    """
    xx, yy = grid
    pos = np.dstack((xx, yy))
    densities = {}

    # Case 1: WGMM-UAPCA (dict of lists of components)
    if isinstance(projection, dict) and all(isinstance(v, list) for v in projection.values()):
        for label, components in projection.items():
            Z_label = np.zeros(xx.shape)
            for mu, cov, w in components:
                rv = multivariate_normal(mean=mu, cov=cov)
                Z_label += w * rv.pdf(pos)
            densities[label] = Z_label

    # Case 2: UAPCA (dict of (mu, cov))
    elif isinstance(projection, dict) and all(isinstance(v, tuple) for v in projection.values()):
        for label, (mu, cov) in projection.items():
            rv = multivariate_normal(mean=mu, cov=cov)
            Z_label = rv.pdf(pos)
            densities[label] = Z_label

    # Case 3: PCA (tuple (X_proj, y))
    elif isinstance(projection, tuple) and len(projection) == 2:
        X_proj, y = projection
        for label in y["Label"].unique():
            label_data = X_proj[y["Label"] == label].values
            kde = gaussian_kde(label_data.T)
            Z_label = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            densities[label] = Z_label

    else:
        raise ValueError("Unsupported projection format for density calculation.")

    return densities