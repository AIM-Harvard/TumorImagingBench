import logging
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["axes.titleweight"] = "bold"


def apply_r_style(fig, font_size=32):
    """Apply a minimal ggplot-like style with larger fonts for readability."""
    fig.update_layout(
        font=dict(size=font_size, family="DejaVu Serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#D0D0D0",
            zeroline=False,
            linecolor="#4B4B4B",
            mirror=True,
            ticks="outside",
            ticklen=6,
            tickwidth=1.5,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#D0D0D0",
            zeroline=False,
            linecolor="#4B4B4B",
            mirror=True,
            ticks="outside",
            ticklen=6,
            tickwidth=1.5,
        ),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _positive_column(probs: np.ndarray) -> np.ndarray:
    """
    Extract the positive-class column for binary problems; otherwise return all columns.
    """
    if probs.shape[1] == 2:
        return probs[:, 1:2]
    return probs


def train_knn_classifier(train_items, train_labels, val_items, val_labels):
    """
    Train a KNN classifier with hyperparameter optimization using Optuna.

    Args:
        train_items: Training feature matrix.
        train_labels: Training labels.
        val_items: Validation feature matrix.
        val_labels: Validation labels.

    Returns:
        best_model: Trained KNeighborsClassifier with the best parameters.
        study: Optuna study object with the optimization results.
    """
    def objective(trial):
        k = trial.suggest_int('k', 1, 50)
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_items, train_labels)
        val_predictions = knn.predict_proba(val_items)
        if val_predictions.shape[1] == 2:  # Binary classification
            return roc_auc_score(val_labels, val_predictions[:, 1])
        else:
            return roc_auc_score(val_labels, val_predictions, multi_class='ovr')

    # Define grid of k values from 1 to 50
    param_grid = {'k': list(range(1, 51))}
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GridSampler(param_grid)
    )
    study.optimize(objective, n_trials=len(param_grid['k']))

    # Train the final model using the best hyperparameter
    best_model = KNeighborsClassifier(n_neighbors=study.best_params['k'], metric='cosine')
    best_model.fit(train_items, train_labels)

    return best_model, study


def evaluate_model(model, test_items, test_labels):
    """
    Evaluate a trained classifier on test data using the ROC AUC score.

    Args:
        model: Trained classifier.
        test_items: Test feature matrix.
        test_labels: Test labels.

    Returns:
        ROC AUC score as a float.
    """
    test_predictions = model.predict_proba(test_items)
    if test_predictions.shape[1] == 2:  # Binary classification
        return roc_auc_score(test_labels, test_predictions[:, 1])
    else:
        return roc_auc_score(test_labels, test_predictions, multi_class='ovr')


def plot_model_comparison(test_accuracies_dict, width=500, height=400, font_size=28, marker_color="#ADD8E6", yshift_annotation=20):
    """
    Create a minimalist and elegant bar plot comparing model performances using Plotly Express.

    Args:
        test_accuracies_dict: Dictionary mapping model names to their performance metrics.
            Each value should be a dict with keys 'mean' and 'ci_95', where 'ci_95' is a tuple (lower_bound, upper_bound).

    Returns:
        Plotly figure object.
    """
    # Extract model names, mean values, and compute error bars for the 95% CI.
    model_names = list(test_accuracies_dict.keys())
    means = [test_accuracies_dict[model]['mean'] for model in model_names]
    error_y = [test_accuracies_dict[model]['ci95'][1] - test_accuracies_dict[model]['mean'] for model in model_names]
    error_y_minus = [test_accuracies_dict[model]['mean'] - test_accuracies_dict[model]['ci95'][0] for model in model_names]

    fig = px.bar(
        x=model_names,
        y=means,
        error_y=error_y,
        error_y_minus=error_y_minus,
        labels={'x': 'Model', 'y': 'AUC'},
        title='',
        template='simple_white',
        width=width,
        height=height
    )
    # Use a subtle blue color and position text inside each bar with minimal formatting;
    # update error bar thickness to make them thicker.
    fig.update_traces(
        marker_color=marker_color,
        marker_opacity=0.8,
        error_y=dict(width=10)
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showline=False, showgrid=True, gridcolor="#D0D0D0", tickfont=dict(size=font_size)),
        yaxis=dict(showgrid=True, gridcolor="#D0D0D0", tickfont=dict(size=font_size)),
        title=dict(x=0.5, xanchor='center'),
        showlegend=False
    )
    
    # Add annotations for mean scores positioned just above the upper confidence interval.
    for model, mean, err in zip(model_names, means, error_y):
        fig.add_annotation(
            x=model,
            y=mean + err,
            text=f"{mean:.2f}",
            showarrow=False,
            yshift=yshift_annotation,
            font=dict(size=font_size - 2, color="black")
        )
    return apply_r_style(fig, font_size=font_size)


def split_shuffle_data(items, labels, train_ratio=0.5, val_ratio=0.2, random_seed=42, stratify=False):
    """
    Split and shuffle data into training, validation, and test sets.

    Args:
        items: Feature array.
        labels: Label array.
        train_ratio: Ratio of data used for training.
        val_ratio: Ratio of data used for validation.
        random_seed: Random seed for reproducibility.
        stratify: If True, stratify splits based on label distribution.

    Returns:
        Tuple containing (train_items, train_labels, val_items, val_labels, test_items, test_labels).
    """
    if stratify:
        # First, split off the test set
        test_ratio = 1 - train_ratio - val_ratio
        train_val_items, test_items, train_val_labels, test_labels = train_test_split(
            items, labels,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=labels
        )
        # Then, split the remaining data into training and validation sets
        train_items, val_items, train_labels, val_labels = train_test_split(
            train_val_items, train_val_labels,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_seed,
            stratify=train_val_labels
        )
    else:
        rng = np.random.default_rng(random_seed)
        shuffled_indices = rng.permutation(len(labels))
        items = items[shuffled_indices]
        labels = np.array(labels)[shuffled_indices]

        train_size = int(train_ratio * len(labels))
        val_size = int(val_ratio * len(labels))

        train_items = items[:train_size]
        train_labels = labels[:train_size]
        val_items = items[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
    return train_items, train_labels, val_items, val_labels, test_items, test_labels


def apply_aggregation_filter(v, model_name):
    if model_name == "MedImageInsightExtractor":
        return v.mean(axis=0)
    elif model_name == "CTClipVitExtractor":
        return v.mean(axis=(1,2,3))
    elif model_name == "PASTAExtractor":
        return v.mean(axis=(2,3,4))        
    else:
        return v

def extract_model_features(data):
    """
    Concatenate features from the train, validation, and test sets for each model.

    Args:
        data (dict): Dictionary where each key is a model name and each value is a dict 
                     with lists for 'train', 'val', and 'test'. Each list contains 
                     dictionaries with a "feature" key.
        skip_model (str): Model name to skip during feature extraction.

    Returns:
        Dictionary mapping model names to a concatenated numpy array of features.
    """
    model_features = {}
    for model_name, splits in data.items():
        features_to_concat = []
        for split in ["train", "val", "test"]:
            if split in splits and splits[split]:
                # Stack features for this split and add to the list.
                split_features = np.vstack([entry["feature"] for entry in splits[split]])
                features_to_concat.append(split_features)
        # Concatenate all split features along axis 0 if any exist; otherwise, use an empty array.
        model_features[model_name] = np.concatenate(features_to_concat, axis=0) if features_to_concat else np.array([])
    return model_features


def compute_knn_indices(model_features, num_neighbors=10, metric="cosine"):
    """
    Compute k-nearest neighbor indices (excluding the sample itself) for each model's features.

    Args:
        model_features (dict): Dictionary mapping model names to feature arrays.
        num_neighbors (int): Number of nearest neighbors to retrieve (excluding self).
        metric (str): Distance metric to use.

    Returns:
        Dictionary mapping model names to an array of nearest neighbor indices.
    """
    model_neighbors = {}
    for model_name, features in model_features.items():
        nn_model = NearestNeighbors(n_neighbors=num_neighbors + 1, metric=metric)
        nn_model.fit(features)
        _, indices = nn_model.kneighbors(features)
        model_neighbors[model_name] = indices[:, 1:]  # Exclude self-neighbor
    return model_neighbors


def compute_overlap_matrix(model_neighbors):
    """
    Compute average mutual k-nearest neighbor overlap scores between pairs of models.

    For each pair of models, this function calculates the average overlap score based on mutual nearest neighbors.
    If the number of samples between models does not match, a warning is issued and that pair is skipped.

    Args:
        model_neighbors (dict): Dictionary mapping model names to arrays of neighbor indices.

    Returns:
        A tuple (overlap_matrix, model_list), where overlap_matrix is a symmetric numpy array of
        average overlap scores and model_list is a list of corresponding model names.
    """
    model_list = list(model_neighbors.keys())
    n_models = len(model_list)
    overlap_matrix = np.full((n_models, n_models), np.nan)

    for i in range(n_models):
        neighbors_a = model_neighbors[model_list[i]]
        for j in range(i + 1, n_models):
            neighbors_b = model_neighbors[model_list[j]]
            if neighbors_a.shape[0] != neighbors_b.shape[0]:
                logging.warning(
                    "Number of samples in %s and %s do not match. Skipping pair.",
                    model_list[i], model_list[j]
                )
                continue

            # Determine overlap using broadcasting
            common_flags = (neighbors_a[:, :, None] == neighbors_b[:, None, :]).any(axis=2)
            sample_overlaps = np.sum(common_flags, axis=1)
            avg_overlap = np.mean(sample_overlaps)

            overlap_matrix[i, j] = avg_overlap
            overlap_matrix[j, i] = avg_overlap
    np.fill_diagonal(overlap_matrix, 0)
    overlap_matrix = np.nan_to_num(overlap_matrix, nan=0.0)
    return overlap_matrix, model_list


def plot_overlap_matrix(overlap_matrix, model_list, title="Mutual k-Nearest Neighbors Overlap Scores", width=1200, height=1200, color="Greens", tickangle=90, font_size=32):
    """
    Plot the mutual k-nearest neighbor overlap matrix using matplotlib (ggplot-like style).

    Args:
        overlap_matrix (numpy.ndarray): Square matrix with average overlap scores.
        model_list (list): List of model names corresponding to the matrix axes.
        title (str): Title of the plot (used as figure title).
        width (int): Width of the plot in pixels.
        tickangle (int): Angle for x-axis tick labels.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    overlap_matrix = np.nan_to_num(overlap_matrix, nan=0.0)
    vmax = float(np.max(overlap_matrix)) if overlap_matrix.size else 1.0
    fig_size = (width / 100, height / 100)  # keep square-ish aspect using pixel->inch conversion

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(overlap_matrix, cmap=color, vmin=0, vmax=max(vmax, 1e-6))
    ax.set_xticks(range(len(model_list)))
    ax.set_xticklabels(model_list, rotation=tickangle, ha="right", fontsize=font_size * 0.6, fontname="DejaVu Serif")
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels(model_list, fontsize=font_size * 0.6, fontname="DejaVu Serif")
    ax.set_title(title, fontsize=font_size, fontname="DejaVu Serif")
    ax.tick_params(length=6, width=1.5, colors="#4B4B4B")
    ax.spines[:].set_visible(True)
    ax.spines[:].set_color("#4B4B4B")
    ax.spines[:].set_linewidth(1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size * 0.6)
    cbar.outline.set_linewidth(1.2)

    fig.tight_layout()
    return fig


def train_linear_probing_classifier(train_items, train_labels, val_items, val_labels):
    """
    Train a linear classifier (logistic regression) on foundation model features.

    Linear probing is a standard evaluation protocol for foundation models where a simple
    linear classifier is trained on frozen feature representations.

    Args:
        train_items: Training feature matrix.
        train_labels: Training labels.
        val_items: Validation feature matrix.
        val_labels: Validation labels.

    Returns:
        model: Trained LogisticRegression classifier.
        score: Validation ROC AUC score.
    """
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42)
    model.fit(train_items, train_labels)

    val_predictions = model.predict_proba(val_items)
    if val_predictions.shape[1] == 2:  # Binary classification
        score = roc_auc_score(val_labels, val_predictions[:, 1])
    else:
        score = roc_auc_score(val_labels, val_predictions, multi_class='ovr')

    return model, score


def train_few_shot_classifier(train_items, train_labels, val_items, val_labels, shots=5, metric='cosine'):
    """
    Train a k-NN classifier with limited training samples (few-shot learning).

    Few-shot learning evaluates how well models generalize with minimal labeled data.

    Args:
        train_items: Training feature matrix.
        train_labels: Training labels.
        val_items: Validation feature matrix.
        val_labels: Validation labels.
        shots: Number of training samples per class to use.
        metric: Distance metric for k-NN.

    Returns:
        model: Trained KNeighborsClassifier with few-shot data.
        score: Validation ROC AUC score.
        selected_indices: Indices of selected training samples.
    """
    # Ensure labels are numpy arrays for advanced indexing
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # Get unique labels and select 'shots' samples per class
    unique_labels = np.unique(train_labels)
    selected_indices = []

    for label in unique_labels:
        label_indices = np.where(train_labels == label)[0]
        if len(label_indices) >= shots:
            selected = np.random.choice(label_indices, shots, replace=False)
        else:
            selected = label_indices
        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)
    few_shot_items = train_items[selected_indices]
    few_shot_labels = train_labels[selected_indices]

    # Train k-NN with optimal k from validation data
    def objective(trial):
        k = trial.suggest_int('k', 1, min(50, len(few_shot_items)))
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(few_shot_items, few_shot_labels)
        val_predictions = knn.predict_proba(val_items)
        if val_predictions.shape[1] == 2:
            return roc_auc_score(val_labels, val_predictions[:, 1])
        else:
            return roc_auc_score(val_labels, val_predictions, multi_class='ovr')

    param_grid = {'k': list(range(1, min(51, len(few_shot_items) + 1)))}
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GridSampler(param_grid)
    )
    study.optimize(objective, n_trials=len(param_grid['k']))

    model = KNeighborsClassifier(n_neighbors=study.best_params['k'], metric=metric)
    model.fit(few_shot_items, few_shot_labels)

    val_predictions = model.predict_proba(val_items)
    if val_predictions.shape[1] == 2:
        score = roc_auc_score(val_labels, val_predictions[:, 1])
    else:
        score = roc_auc_score(val_labels, val_predictions, multi_class='ovr')

    return model, score, selected_indices


def build_knn_ensemble_classifier(train_items_dict, train_labels, val_items_dict, val_labels, overlap_matrix, model_list, k=10):
    """
    Build an ensemble classifier using mutual k-NN alignment to weight model contributions.

    The intuition: models with high mutual k-NN overlap have similar embedding spaces and
    thus similar inductive biases. Models with higher mutual overlap scores are weighted more
    heavily in the ensemble prediction.

    Args:
        train_items_dict: Dictionary mapping model names to training feature matrices.
        train_labels: Training labels (shared across all models).
        val_items_dict: Dictionary mapping model names to validation feature matrices.
        val_labels: Validation labels.
        overlap_matrix: k-NN overlap matrix between models (from compute_overlap_matrix).
        model_list: List of model names corresponding to overlap matrix axes.
        k: Number of neighbors for k-NN.

    Returns:
        ensemble_model: Dictionary containing trained k-NN models for each model and weights.
        score: Validation ROC AUC score of the ensemble.
    """
    # Train individual k-NN classifiers for each model
    individual_models = {}
    for model_name in model_list:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_items_dict[model_name], train_labels)
        individual_models[model_name] = knn

    # Compute model weights based on overlap matrix (higher overlap = higher weight)
    # Fill diagonal with 1.0 (self-overlap) and compute average overlap for each model
    np.fill_diagonal(overlap_matrix, 1.0)
    model_weights = np.nanmean(overlap_matrix, axis=1)
    model_weights = model_weights / np.nansum(model_weights)  # Normalize to sum to 1

    weights_dict = {name: weight for name, weight in zip(model_list, model_weights)}

    # Evaluate ensemble on validation set
    ensemble_predictions = np.zeros((len(val_labels), len(np.unique(train_labels))))
    for model_name in model_list:
        preds = individual_models[model_name].predict_proba(val_items_dict[model_name])
        ensemble_predictions += weights_dict[model_name] * preds

    # Normalize predictions
    ensemble_predictions = ensemble_predictions / ensemble_predictions.sum(axis=1, keepdims=True)

    if ensemble_predictions.shape[1] == 2:
        score = roc_auc_score(val_labels, ensemble_predictions[:, 1])
    else:
        score = roc_auc_score(val_labels, ensemble_predictions, multi_class='ovr')

    ensemble_model = {
        'individual_models': individual_models,
        'weights': weights_dict,
        'model_list': model_list
    }

    return ensemble_model, score


def predict_with_ensemble(ensemble_model, test_items_dict, model_list):
    """
    Make predictions using the k-NN ensemble model.

    Args:
        ensemble_model: Ensemble model dictionary from build_knn_ensemble_classifier.
        test_items_dict: Dictionary mapping model names to test feature matrices.
        model_list: List of model names.

    Returns:
        ensemble_predictions: Weighted ensemble predictions.
    """
    individual_models = ensemble_model['individual_models']
    weights = ensemble_model['weights']

    # Get number of classes from first model's predictions
    first_pred = individual_models[model_list[0]].predict_proba(test_items_dict[model_list[0]])
    n_classes = first_pred.shape[1]

    ensemble_predictions = np.zeros((len(test_items_dict[model_list[0]]), n_classes))

    for model_name in model_list:
        preds = individual_models[model_name].predict_proba(test_items_dict[model_name])
        ensemble_predictions += weights[model_name] * preds

    # Normalize predictions
    ensemble_predictions = ensemble_predictions / ensemble_predictions.sum(axis=1, keepdims=True)

    return ensemble_predictions


def train_stacking_ensemble_classifier(
    train_items_dict,
    train_labels,
    val_items_dict,
    val_labels,
    k_candidates=(5, 10, 15, 25),
    meta_C_candidates=(0.25, 1.0, 4.0),
    metric="cosine",
):
    """
    Train a stacking ensemble with a logistic regression meta-learner on top of k-NN bases.

    Steps:
        1) Fit a k-NN per model (optionally sweeping k_candidates).
        2) Use validation probabilities from each base as meta-features.
        3) Fit a logistic regression meta-learner with light regularization search.

    Args:
        train_items_dict: Dict of model name -> training feature matrix.
        train_labels: Training labels.
        val_items_dict: Dict of model name -> validation feature matrix.
        val_labels: Validation labels.
        k_candidates: Iterable of neighbor counts to try for base k-NN models.
        meta_C_candidates: Iterable of inverse regularization strengths for the meta-learner.
        metric: Distance metric for k-NN.

    Returns:
        ensemble_model: Dict with base models, meta model, and model list.
        best_score: Best validation ROC AUC achieved.
    """
    model_list = list(train_items_dict.keys())
    best_score = -np.inf
    best_model = None
    n_classes = len(np.unique(train_labels))

    for k in k_candidates:
        # Train base models for this k and cache validation meta-features
        base_models = {}
        val_meta_features = []
        for model_name in model_list:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(train_items_dict[model_name], train_labels)
            base_models[model_name] = knn

            val_probs = knn.predict_proba(val_items_dict[model_name])
            val_meta_features.append(_positive_column(val_probs))

        val_meta = np.hstack(val_meta_features)

        # Tune meta-learner strength on top of fixed base features
        for meta_C in meta_C_candidates:
            meta_model = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                multi_class="ovr" if n_classes > 2 else "auto",
                class_weight="balanced",
                C=meta_C,
            )
            meta_model.fit(val_meta, val_labels)
            val_pred = meta_model.predict_proba(val_meta)
            if val_pred.shape[1] == 2:
                score = roc_auc_score(val_labels, val_pred[:, 1])
            else:
                score = roc_auc_score(val_labels, val_pred, multi_class="ovr")

            if score > best_score:
                best_score = score
                best_model = {
                    "base_models": base_models,
                    "meta_model": meta_model,
                    "model_list": model_list,
                    "k": k,
                    "meta_C": meta_C,
                }

    return best_model, best_score


def predict_with_stacking_ensemble(ensemble_model, items_dict):
    """
    Predict using a trained stacking ensemble.

    Args:
        ensemble_model: Output of train_stacking_ensemble_classifier.
        items_dict: Dict of model name -> feature matrix for the desired split.

    Returns:
        Ensemble probability predictions.
    """
    base_models = ensemble_model["base_models"]
    model_list = ensemble_model["model_list"]
    meta_model = ensemble_model["meta_model"]

    meta_features = []
    for model_name in model_list:
        probs = base_models[model_name].predict_proba(items_dict[model_name])
        meta_features.append(_positive_column(probs))

    stacked = np.hstack(meta_features)
    return meta_model.predict_proba(stacked)
