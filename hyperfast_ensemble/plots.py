from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_vs_pcs(pcs: list[int], accuracies: list[float], title: str = "Accuracy vs. #PCs"):
    plt.figure()
    plt.plot(pcs, accuracies, marker="o")
    plt.xlabel("# Principal Components")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def scatter_tsne(tsne_2d: np.ndarray, labels: np.ndarray, title: str = "t-SNE (2D)"):
    plt.figure()
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], s=8, label=str(cls), alpha=0.7)
    plt.legend(markerscale=2, fontsize="small", ncol=3)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()
