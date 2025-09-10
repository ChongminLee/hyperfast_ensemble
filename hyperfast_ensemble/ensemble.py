from __future__ import annotations
import numpy as np

def majority_vote(preds: list) -> np.ndarray:
    """Hard-vote majority over a list of integer prediction arrays of shape (n_samples,).
    Ties are broken by the first model in the list.
    """
    preds = np.vstack(preds)  # (n_models, n_samples)
    # For each sample, pick the most frequent label
    out = []
    for i in range(preds.shape[1]):
        labels, counts = np.unique(preds[:, i], return_counts=True)
        # tie-break by earliest appearance in preds order
        max_count = counts.max()
        candidates = labels[counts == max_count]
        if len(candidates) == 1:
            out.append(candidates[0])
        else:
            # pick the candidate predicted by the first model among the tied
            first_choice = preds[0, i]
            out.append(first_choice if first_choice in candidates else candidates[0])
    return np.array(out)
