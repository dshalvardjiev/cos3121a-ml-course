"""Plots and metric reports used across sessions (matplotlib only, no seaborn)."""
import numpy as np
import matplotlib.pyplot as plt


def metric_report(y_true, y_prob, threshold: float = 0.5, name: str = "model") -> dict:
    """Print + return the Session 2 metric set for a binary classifier."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
    }
    print(f"--- {name} ---")
    for k, v in m.items():
        print(f"{k:>10}: {v:.3f}")
    return m


def plot_confusion(y_true, y_prob, threshold: float = 0.5, labels=("no churn", "churn")):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix")
    plt.tight_layout(); plt.show()
    return cm


def plot_clusters_2d(X_scaled, cluster_labels, title="Customer segments (PCA projection)"):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pts = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=cluster_labels, cmap="tab10", s=12, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title(title); plt.colorbar(sc, label="cluster"); plt.tight_layout(); plt.show()
    return pca


def plot_training(rewards, window: int = 50, title="RL training progress"):
    r = np.asarray(rewards, dtype=float)
    smooth = np.convolve(r, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(r, alpha=0.25, label="episode reward")
    ax.plot(range(window - 1, len(r)), smooth, lw=2, label=f"rolling mean ({window})")
    ax.set_xlabel("episode"); ax.set_ylabel("reward"); ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()
