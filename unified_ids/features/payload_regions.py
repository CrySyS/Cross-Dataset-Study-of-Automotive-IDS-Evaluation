
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class CLAIndexer:
    """Per-ID KMeans centroids + min/max radius bounds."""
    def __init__(self, k: int = 300, random_state: int = 0):
        self.k = k
        self.random_state = random_state
        self.models: Dict[int, KMeans] = {}
        self.bounds: Dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame):
        # Extract payload bytes from data0-data7 columns
        data_cols = [f"data{i}" for i in range(8)]
        for cid, grp in df.groupby("can_id"):
            # Stack data0-data7 into 8D vectors
            X = grp[data_cols].values.astype(float)
            km = KMeans(n_clusters=min(self.k, len(X)), random_state=self.random_state, n_init=5)
            km.fit(X)
            self.models[cid] = km
            # min/max distance bounds per centroid
            labels = km.labels_
            d = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
            mins = np.zeros(km.n_clusters)
            maxs = np.zeros(km.n_clusters)
            for i in range(km.n_clusters):
                di = d[labels == i]
                if len(di) > 0:
                    mins[i] = np.percentile(di, 1)
                    maxs[i] = np.percentile(di, 99)
            self.bounds[cid] = (mins, maxs)
        return self

    def transform_symbol(self, cid: int, payload) -> tuple[int, float]:
        """Transform payload to (cluster_id, distance).
        
        Args:
            cid: CAN ID
            payload: either list/array of 8 bytes or single row as array
        """
        if cid not in self.models:
            return (-1, float("inf"))
        x = np.array(payload, dtype=float).ravel()[:8]  # ensure 8D
        if len(x) < 8:
            x = np.pad(x, (0, 8 - len(x)), constant_values=0)
        km = self.models[cid]
        i = int(km.predict(x.reshape(1, -1))[0])
        d = float(np.linalg.norm(x - km.cluster_centers_[i]))
        return (i, d)
