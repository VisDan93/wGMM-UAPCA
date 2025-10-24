import numpy as np
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from ipywidgets import FloatSlider, Checkbox, Button, Output, HTML, HBox, VBox, Layout
from IPython.display import display, clear_output

plot_colors = [
    "#106FBC",  # blue
    "#FF7F00",  # orange
    "#4DAF4A",  # green
    "#CE38E5",  # purple
    "#E41A1C",  # red
    "#6D23B6",  # dark purple
    "#85D4FF",  # light blue
    "#A65628",  # brown
    "#FFD700",  # gold
    "#4B4B4B",  # dark gray
    "#00CED1",  # dark turquoise
    "#FF69B4",  # hot pink
    "#8B4513",  # saddle brown
    "#7FFF00",  # chartreuse green
    "#FF4500",  # orange-red
    "#20B2AA",  # light sea green
    "#9932CC",  # dark orchid
    "#FF1493",  # deep pink
    "#4682B4",  # steel blue
    "#9ACD32"   # yellow-green
]


def plot_contours(ax: Axes, xx: np.ndarray, yy: np.ndarray, densities: dict[int, np.ndarray],
                  unique_labels: list[int]) -> list[Line2D]:
    """
    Plot 2D density contours for each label on the given axes and return a legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    xx, yy : np.ndarray
        Meshgrid arrays for the plot coordinates.
    densities : dict[int, np.ndarray]
        A dictionary mapping each label to its 2D density grid.
    unique_labels : list[int]
        A sorted list of unique labels.

    Returns
    -------
    legend_handles : list[matplotlib.lines.Line2D]
        A list of Line2D objects representing legend entries.
    """
    legend_handles = []

    for i, label in enumerate(unique_labels):
        z = densities[label]

        # Normalize density
        z_sum = np.sum(z)
        if z_sum > 0:
            z = z / z_sum

        # Compute contour thresholds
        sorted_z = np.sort(z.flatten())[::-1]
        cumulative_prob = np.cumsum(sorted_z)
        levels = [0.25, 0.5, 0.95]
        thresholds = []
        for level in levels:
            idx = np.searchsorted(cumulative_prob, level)
            if idx < len(sorted_z):
                thresholds.append(sorted_z[idx])
        
        # Plot contours with alpha levels
        for j, threshold in enumerate(sorted(thresholds)):
            alpha = levels[j]
            ax.contour(xx, yy, z, levels=[threshold, threshold + 1e-10],
                       colors=[plot_colors[i]], linewidths=1.2, alpha=alpha)

        # Prepare legend handle
        label_text = f'Label {label}'
        handle = Line2D([0], [0], color=plot_colors[i],
                        linewidth=2, label=label_text)
        legend_handles.append(handle)

    return legend_handles


def plot_and_save_projections(dataset_name: str, X_proj: np.ndarray, y: pd.DataFrame, densities: dict[str, dict[int, np.ndarray]], grid: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Plot 2D projected samples with density contours and save figures per approach.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (used for folder and plot title).
    X_proj : np.ndarray
        Projected 2D samples (n_samples x 2).
    y : pd.DataFrame
        Label DataFrame with column "Label".
    densities : dict[str, dict[int, np.ndarray]]
        Dictionary mapping method name -> (label -> 2D density grid).
    grid : tuple[np.ndarray, np.ndarray]
        Meshgrid arrays (xx, yy) for the plot coordinates.
    weights : dict[int, float]
        Dictionary mapping label -> current weight.
    legend : bool, optional
        Whether to display the legend (default is True).
    """
    xx, yy = grid
    unique_labels = sorted(y["Label"].unique())
    
    save_path = os.path.join("./results", dataset_name)
    os.makedirs(save_path, exist_ok=True)

    for method_name, method_densities in densities.items():
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot contours
        plot_contours(ax, xx, yy, method_densities, unique_labels)

        # Scatter projected points
        for i, label in enumerate(unique_labels):
            mask = y["Label"] == label
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                s=4, alpha=0.5, color=plot_colors[i],
                edgecolors="none",
                label=f"Label {label}"
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0)
        plt.tight_layout()

        # Construct filename
        method_safe = method_name.lower().replace(" ", "-")
        filename = f"{dataset_name}_{method_safe}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)


def align_projection(P: np.ndarray, P_ref: np.ndarray) -> np.ndarray:
    """
    Align projection matrix P to reference projection P_ref.
    
    Parameters
    ----------
    P : np.ndarray
        Projection matrix to be aligned (n_features x n_dims).
    P_ref : np.ndarray
        Reference projection matrix (n_features x n_dims).
    
    Returns
    -------
    P_aligned : np.ndarray
        Aligned projection matrix.
    """
    P = P.copy()
    # Normalize columns
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    P_ref /= np.linalg.norm(P_ref, axis=0, keepdims=True)

    # Ensure same direction for each eigenvector
    for j in range(P.shape[1]):
        if np.dot(P_ref[:, j], P[:, j]) < 0:
            P[:, j] *= -1

    # Ensure right-handed coordinate system
    if np.linalg.det(P_ref.T @ P) < 0:
        P[:, 1] *= -1

    return P


def _normalize_weights(sliders: dict[str, FloatSlider], changed_label: str, new_value: float, unique_labels: list[str], n_labels: int, auto_update: Checkbox, update_plot: callable, updating_flag: dict[str, bool]) -> None:
    """
    Normalize all slider weights so that their total remains equal to 1.0.

    This function updates the slider values whenever one slider is changed.
    The changed slider receives its new value, and the remaining sliders are
    proportionally adjusted to maintain a total sum of 1. It also prevents
    recursive updates via an updating flag.

    Parameters
    ----------
    sliders : dict[str, FloatSlider]
        Dictionary mapping label -> corresponding weight slider widget.
    changed_label : str
        The label whose slider value was changed by the user.
    new_value : float
        The new value assigned to the changed slider.
    unique_labels : list[str]
        List of all unique class labels.
    n_labels : int
        Total number of labels.
    auto_update : Checkbox
        Checkbox indicating whether the plot should auto-update after changes.
    update_plot : callable
        Function to update the plot visualization.
    updating_flag : dict[str, bool]
        Mutable flag used to prevent recursive slider updates.
    """
    # Only normalize if auto-update is enabled
    if not auto_update.value:
        return
    
    # Prevent re-entrant calls while sliders are being programmatically updated
    if updating_flag["value"]:
        return
    updating_flag["value"] = True

    # Extract current slider values and update the changed one
    vals = np.array([sliders[label].value for label in unique_labels], dtype=float)
    idx = unique_labels.index(changed_label)
    vals[idx] = float(new_value)

    # Distribute remaining weight proportionally among other sliders
    remaining = 1.0 - vals[idx]
    other_idx = [i for i in range(n_labels) if i != idx]
    if other_idx:
        sum_other_old = vals[other_idx].sum()
        if sum_other_old > 0:
            vals[other_idx] *= remaining / sum_other_old
        else:
            # If others were zero, distribute remaining equally
            vals[other_idx] = remaining / len(other_idx)

    # Update all sliders with normalized values
    for i, label in enumerate(unique_labels):
        sliders[label].value = float(vals[i])

    # Unlock flag and optionally trigger live plot update
    updating_flag["value"] = False
    if auto_update.value:
        update_plot()


def _make_controls(unique_labels: list[str], sample_based: dict[str, float], update_plot: callable, sliders: dict[str, FloatSlider], updating_flag: dict[str, bool]) -> VBox:
    """
    Create interactive control widgets for weight adjustment and plot updating.

    Includes buttons for equal-weight and sample-based presets,
    as well as options for manual or automatic plot updates.

    Parameters
    ----------
    unique_labels : list[str]
        List of unique labels.
    sample_based : dict[str, float]
        Sample-proportional weights for each label.
    update_plot : callable
        Callback to trigger the plot update.
    sliders : dict[str, FloatSlider]
        Mapping of label -> slider widget for weight control.
    updating_flag : dict[str, bool]
        Flag used to prevent redundant slider updates.

    Returns
    -------
    controls : VBox
        A vertical container with the interactive control panel.
    """
    n_labels = len(unique_labels)

    # Main control widgets
    auto_update = Checkbox(value=True, description="Auto Update Plot", indent=False, layout=Layout(width="180px"))
    update_button = Button(description="Update Plot", button_style="primary", layout=Layout(width="180px", height="36px"))
    equal_button = Button(description="Equal", button_style="info", layout=Layout(width="180px", height="36px"))
    sample_button = Button(description="Sample-Based", button_style="info", layout=Layout(width="180px", height="36px"))

    # Button callbacks
    def on_update_clicked(_):
        # Normalize all weights first
        total = sum(sliders[lbl].value for lbl in unique_labels)
        if total > 0:
            for lbl in unique_labels:
                sliders[lbl].value /= total
        update_plot()

    def on_equal_clicked(_):
        # Set all sliders to equal weights
        updating_flag["value"] = True
        for lbl in unique_labels:
            sliders[lbl].value = 1.0 / n_labels
        updating_flag["value"] = False
        update_plot()

    def on_sample_clicked(_):
        # Set all sliders to sample-based weights
        updating_flag["value"] = True
        arr = np.array([sample_based.get(label, 0.0) for label in unique_labels], dtype=float)
        arr /= arr.sum() if arr.sum() > 0 else len(arr)
        for i, lbl in enumerate(unique_labels):
            sliders[lbl].value = float(arr[i])
        updating_flag["value"] = False
        update_plot()

    # Attach callbacks to buttons
    update_button.on_click(on_update_clicked)
    equal_button.on_click(on_equal_clicked)
    sample_button.on_click(on_sample_clicked)

    # Layout organization
    quick_label = HTML("<h3 style='color:#007acc; font-size:20px; margin:0 0 10px 5px;'>Weight Presets</h3>")
    first_row = VBox(
        [quick_label, HBox([equal_button, sample_button], layout=Layout(gap='25px'))],
        layout=Layout(padding="5px 0")
    )

    second_row = HBox(
        [update_button, auto_update],
        layout=Layout(gap="40px", margin="20px 0 0 0", align_items="center")
    )

    return VBox([first_row, second_row], layout=Layout(margin="20px"))


def _update_plot(X: pd.DataFrame, y: pd.DataFrame, gmm: dict[str, GaussianMixture], sliders: dict[str, FloatSlider], unique_labels: list[str], output: Output, P_ref_container: dict[str, np.ndarray]) -> None:
    """
    Redraw the interactive visualization with updated mixture weights.

    This function computes the weighted projection using WGMM-UAPCA,
    calculates density contours, and re-renders the scatter plot of the
    projected dataset with overlaid Gaussian mixture contours.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of the dataset.
    y : pd.DataFrame
        Label DataFrame with a column 'Label'.
    gmm : dict[str, GaussianMixture]
        Dictionary of label -> fitted Gaussian Mixture Model.
    sliders : dict[str, FloatSlider]
        Mapping of label -> current slider widgets controlling weights.
    unique_labels : list[str]
        List of all unique labels.
    output : ipywidgets.Output
        Output widget for rendering Matplotlib plots in Jupyter.
    P_ref_container : dict[str, np.ndarray]
        Container holding the reference projection matrix for alignment.
    """
    import wgmm_uapca as wgmm

    with output:
        clear_output(wait=True)

        # Retrieve current slider weights and normalize
        weights_array = np.array([sliders[label].value for label in unique_labels], dtype=float)
        weights_array /= weights_array.sum() if weights_array.sum() > 0 else len(weights_array)

        labels_series = y["Label"]

        # Compute 2D projection matrix and project the dataset
        P = wgmm.calculate_projmat(gmm, weights=weights_array, n_dims=2)

        # Align new projection to previous one for visual stability
        P_ref = P_ref_container.get("P", None)
        if P_ref is not None:
            P = align_projection(P, P_ref)
        P_ref_container["P"] = P

        # Project data
        X_proj = X.values @ P

        # Generate grid and projected GMM for density evaluation
        x_coords, y_coords = wgmm.calculate_grid(gmm, P)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid = (xx, yy)

        proj_wgmm = wgmm.wgmm_uapca(gmm, P)
        densities = wgmm.calculate_density(proj_wgmm, grid)

        # Plot 2D densities and scatter data
        _, ax = plt.subplots(figsize=(8, 6))
        plot_contours(ax, xx, yy, densities, unique_labels)

        for i, label in enumerate(unique_labels):
            mask = labels_series == label
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                s=8, alpha=0.55,
                color=wgmm.plot_colors[i],
                edgecolors="none"
            )

        # Clean up plot aesthetics
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()


def create_interactive_plot(dataset_name: str, dataset: tuple[pd.DataFrame, pd.DataFrame], gmm: dict[str, GaussianMixture]) -> None:
    """
    Create and display an interactive widget for weighted GMM visualization.

    This interface allows dynamic adjustment of label-specific weights
    via sliders, with real-time (or manual) updates to a 2D WGMM-UAPCA plot.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (used for titles).
    dataset : tuple[pd.DataFrame, pd.DataFrame]
        Tuple (X, y) where X is feature data and y contains label information.
    gmm : dict[str, GaussianMixture]
        Dictionary of label -> trained Gaussian Mixture Model.
    """
    X, y = dataset
    labels_series = y["Label"]

    # Extract label and sample information
    unique_labels = sorted(list(gmm.keys()))
    n_labels = len(unique_labels)
    label_counts = labels_series.value_counts(normalize=True).to_dict()
    sample_based = {label: float(label_counts.get(label, 0.0)) for label in unique_labels}

    # Create individual sliders for each label
    sliders = {
        label: FloatSlider(
            value=1.0 / n_labels,
            min=0.01, max=0.98, step=0.01,
            description=f"Label {label}",
            continuous_update=False,
            readout_format=".2f",
            layout=Layout(width="420px", height="32px"),
            style={'description_width': '90px'}
        )
        for label in unique_labels
    }

    output = Output()
    updating_flag = {"value": False}

    P_ref_container = {"P": None}

    # Plot update wrapper
    def update_plot(*_):
        _update_plot(X, y, gmm, sliders, unique_labels, output, P_ref_container)

    # Create control panel
    controls = _make_controls(unique_labels, sample_based, update_plot, sliders, updating_flag)
    auto_update = controls.children[1].children[1]

    # Connect slider events for live normalization
    for lbl in unique_labels:
        sliders[lbl].observe(
            lambda change, lbl=lbl: _normalize_weights(
                sliders, lbl, change["new"], unique_labels, n_labels, auto_update, update_plot, updating_flag
            ),
            names="value",
        )

    slider_label = HTML("<h3 style='color:#007acc; font-size:20px; margin: 0 0 10px 50px;'>Weight Adjustment</h3>")

    if n_labels >= 8:
        half = (n_labels + 1) // 2  # round up
        left_labels = list(sliders.keys())[:half]
        right_labels = list(sliders.keys())[half:]

        left_column = VBox([sliders[lbl] for lbl in left_labels], layout=Layout(gap="15px"))
        right_column = VBox([sliders[lbl] for lbl in right_labels], layout=Layout(gap="15px"))
        sliders_box = VBox([slider_label, HBox([left_column, right_column], layout=Layout(gap="30px"))],
                           layout=Layout(margin="25px 0 0 0"))
    else:
        # Single column if fewer than 8 labels
        left_column = VBox([sliders[lbl] for lbl in sliders.keys()], layout=Layout(gap="15px"))
        sliders_box = VBox([slider_label, left_column], layout=Layout(margin="25px 0 0 0"))

    # Right panel: controls
    right_panel = VBox([controls], layout=Layout(align_items="flex-start"))

    header_html = f"""
    <h2 style='font-size:30px; font-weight:800;
                margin:10px 0 10px 0; letter-spacing:0.5px;
                text-align:center; line-height:1.3;'>
          Interactive Weight Visualization for <br>
          <span style='color:#007acc;'>{dataset_name.title()}</span> Dataset
    </h2>
    """
    header = HTML(header_html)

    # Assemble final layout and render
    layout_box = VBox(
        [
            header,
            HBox(
                [sliders_box, right_panel],
                layout=Layout(justify_content="center", align_items="flex-start", width="100%")
            ),
            output
        ],
        layout=Layout(align_items="center", justify_content="center", width="100%", overflow="visible")
    )

    display(layout_box)
    update_plot()