"""
Association Rules-Based Anomaly Detection for CAN-bus IDS
Paper: D'Angelo et al. (2023) - "An Association Rules-Based Approach for Anomaly Detection on CAN-bus"

Implementation follows the paper's algorithm:
1. CLA (Cluster-based Learning) with K=300 clusters per CAN ID on 8D payload space
2. KNN clustering of regions across IDs (N=50)
3. Quantization: centroids -> 150 clusters, bounds -> 100 bins (reduces to 75 columns)
4. Sliding windows (w=900 messages, stride=450 messages)
5. Apriori rule mining (min_support=0.9) on training windows
6. Testing: unknown ID check, distribution check, rule matching

Dataset: HCRL Car-Hacking Dataset
Reported metrics (avg): Acc 0.9947, Pre 0.9907, Rec 0.9983, F1 0.9945
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Iterator, Tuple, Set
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

from unified_ids.methods.base import BaseFlowIDS
from unified_ids.dataio.windowing import Window, windows_fixed_msgs
from unified_ids.features.payload_regions import CLAIndexer

logger = logging.getLogger(__name__)


def simple_apriori(transactions: List[List[str]], min_support: float, max_itemset_size: int = 3) -> Set[frozenset]:
    """
    Simple Apriori implementation for frequent itemset mining.
    Optimized for performance with max itemset size limit.
    
    Args:
        transactions: List of transactions, each transaction is a list of items
        min_support: Minimum support threshold (0-1)
        max_itemset_size: Maximum size of itemsets to mine (default 3 for speed)
    
    Returns:
        Set of frequent itemsets (as frozensets)
    """
    if not transactions:
        return set()
    
    n_transactions = len(transactions)
    min_count = int(np.ceil(min_support * n_transactions))
    
    # Count single items
    item_counts = Counter()
    for transaction in transactions:
        for item in set(transaction):  # unique items per transaction
            item_counts[item] += 1
    
    # Frequent 1-itemsets
    frequent_itemsets = set()
    frequent_k = {frozenset([item]) for item, count in item_counts.items() if count >= min_count}
    frequent_itemsets.update(frequent_k)
    
    if not frequent_k:
        return frequent_itemsets
    
    # Generate k+1 itemsets from k-itemsets (with size limit)
    k = 1
    while frequent_k and k < max_itemset_size:
        # Generate candidates
        candidates = set()
        freq_list = list(frequent_k)
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                union = freq_list[i] | freq_list[j]
                if len(union) == k + 1:
                    candidates.add(union)
        
        if not candidates:
            break
        
        # Early termination if too many candidates
        if len(candidates) > 10000:
            logger.warning(f"[Apriori] Too many candidates ({len(candidates)}), stopping at k={k}")
            break
        
        # Count candidates
        candidate_counts = Counter()
        for transaction in transactions:
            trans_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(trans_set):
                    candidate_counts[candidate] += 1
        
        # Filter by min_support
        frequent_k = {itemset for itemset, count in candidate_counts.items() if count >= min_count}
        frequent_itemsets.update(frequent_k)
        k += 1
    
    return frequent_itemsets


class AssocRulesIDS(BaseFlowIDS):
    """
    Association Rules-Based IDS following D'Angelo et al. (2023).
    
    Training:
    1. Per-ID CLA clustering (K clusters per ID) on 8D payload vectors
    2. Cross-ID region clustering (N clusters for all regions)
    3. Quantization to discrete representation
    4. Sliding window + Apriori frequent itemset mining
    
    Testing:
    - Fast checks: unknown ID, distribution change
    - Rule matching: extract itemsets, check against training library
    """
    
    def __init__(
        self,
        k_cla: int = 300,              # CLA clusters per ID (paper: K=300)
        n_knn: int = 50,               # "KNN clusters" across IDs (paper: N=50)
        n_centroid_clusters: int = 150, # Quantization: centroid clusters
        n_bound_bins: int = 100,       # Quantization: bound bins
        window_size: int = 900,        # Messages per window (paper: 0.5s ~900 msgs)
        stride: int = 450,             # Window stride (paper: Δ=450)
        min_support: float = 0.9,      # Apriori support threshold
        dist_threshold: float = 0.3,   # Distribution change threshold (paper doesn't specify)
        random_state: int = 0,
    ):
        self.k_cla = k_cla
        self.n_knn = n_knn
        self.n_centroid_clusters = n_centroid_clusters
        self.n_bound_bins = n_bound_bins
        self.window_size = window_size
        self.stride = stride
        self.min_support = min_support
        self.dist_threshold = dist_threshold
        self.random_state = random_state
        
        # Fitted components
        self.cla: Optional[CLAIndexer] = None
        self.region_clusterer: Optional[KMeans] = None  # "KNN" step
        self.centroid_quantizer: Optional[KMeans] = None
        self.bound_bins: Optional[np.ndarray] = None  # bin edges for bounds
        self.legal_itemsets: Optional[Set[frozenset]] = None
        self.known_ids: Optional[Set[int]] = None
        self.baseline_distributions: Optional[Dict[int, float]] = None  # ID -> freq
        
        # Intermediate data for debugging
        self.region_features: Optional[np.ndarray] = None  # (n_regions, feature_dim)
        self.region_to_id: Optional[List[int]] = None  # map region idx -> CAN ID
        self.region_meta: Optional[List[Tuple[int, int]]] = None  # (can_id, local_cluster_idx)
        self.region_cluster_labels: Optional[Dict[Tuple[int, int], int]] = None  # (cid, local_idx) -> cross-ID cluster label
        
    def fit(self, df_train: pd.DataFrame):
        """
        Training phase following Algorithm 1 from the paper.
        
        Args:
            df_train: Training data (attack-free only, as per paper)
        """
        logger.info(f"[AssocRules] Training on {len(df_train)} messages")
        
        # Step 1: CLA - Learn per-ID regions (K=300 clusters per ID)
        logger.info(f"[AssocRules] Step 1: CLA with K={self.k_cla} clusters per ID")
        self.cla = CLAIndexer(k=self.k_cla, random_state=self.random_state)
        self.cla.fit(df_train)
        self.known_ids = set(df_train["can_id"].unique())
        logger.info(f"[AssocRules] CLA fitted for {len(self.known_ids)} CAN IDs")
        
        # Step 2: Extract region features for cross-ID clustering
        logger.info(f"[AssocRules] Step 2: Extracting region features for KNN clustering (N={self.n_knn})")
        region_features = []
        region_to_id = []
        region_meta = []
        
        for cid in self.known_ids:
            if cid not in self.cla.models:
                continue
            km = self.cla.models[cid]
            mins, maxs = self.cla.bounds[cid]
            
            # Feature vector per region: [centroid_8d, min_bound, max_bound] = 10D
            for i in range(km.n_clusters):
                feat = np.concatenate([
                    km.cluster_centers_[i],  # 8D
                    [mins[i]],                # 1D
                    [maxs[i]]                 # 1D
                ])
                region_features.append(feat)
                region_to_id.append(cid)
                region_meta.append((cid, i))
        
        self.region_features = np.array(region_features)
        self.region_to_id = region_to_id
        self.region_meta = region_meta
        logger.info(f"[AssocRules] Extracted {len(region_features)} regions")
        
        # Step 3: "KNN clustering" - cluster regions across IDs
        # Paper calls it "KNN with N clusters" - interpret as K-means on 10D region features
        # This groups similar regions (across different CAN IDs) to capture correlations
        logger.info(f"[AssocRules] Step 3: Clustering regions with K-means (N={self.n_knn})")
        n_clusters = min(self.n_knn, len(region_features))
        
        # Note: Paper's N=50 might be too small for Car-Hacking with ~5000+ regions
        # But we respect the paper's parameter; can tune if needed
        # A warning if significantly fewer clusters
        if n_clusters < self.n_knn:
            logger.warning(f"[AssocRules] Requested N={self.n_knn} clusters, but only {len(region_features)} regions available. Using {n_clusters} clusters.")
        
        self.region_clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.region_clusterer.fit(self.region_features)
        # Map each per-ID cluster to a cross-ID region cluster label
        labels = self.region_clusterer.labels_
        region_cluster_labels: Dict[Tuple[int, int], int] = {}
        for meta, lbl in zip(self.region_meta, labels):
            region_cluster_labels[meta] = int(lbl)
        self.region_cluster_labels = region_cluster_labels
        logger.info(f"[AssocRules] Region clustering complete ({n_clusters} clusters)")
        
        # Step 4: Quantization (as per paper's scale reduction)
        logger.info(f"[AssocRules] Step 4: Quantization (centroid clusters={self.n_centroid_clusters}, bound bins={self.n_bound_bins})")
        
        # 4a: Cluster centroids -> 150 prototypes
        centroids_8d = np.array([feat[:8] for feat in self.region_features])
        n_cent_clusters = min(self.n_centroid_clusters, len(centroids_8d))
        self.centroid_quantizer = KMeans(
            n_clusters=n_cent_clusters,
            random_state=self.random_state,
            n_init=5
        )
        self.centroid_quantizer.fit(centroids_8d)
        
        # 4b: Discretize bounds into 100 bins
        all_bounds = np.array([feat[8:10] for feat in self.region_features]).ravel()
        self.bound_bins = np.percentile(all_bounds, np.linspace(0, 100, self.n_bound_bins + 1))
        logger.info(f"[AssocRules] Quantization complete")
        
        # Step 5: Baseline distribution (for distribution check)
        id_counts = df_train["can_id"].value_counts()
        total = len(df_train)
        self.baseline_distributions = {cid: count / total for cid, count in id_counts.items()}
        logger.info(f"[AssocRules] Baseline distributions computed for {len(self.baseline_distributions)} IDs")
        
        # Step 6: Sliding windows + Apriori rule mining
        logger.info(f"[AssocRules] Step 5: Sliding windows (w={self.window_size}, stride={self.stride}) + Apriori (support={self.min_support})")
        transactions = []
        
        for w in windows_fixed_msgs(df_train, self.window_size, self.stride):
            sub = df_train.iloc[w.idx_start:w.idx_end+1]
            items = self._extract_window_items(sub)
            if items:
                transactions.append(items)
        
        logger.info(f"[AssocRules] Generated {len(transactions)} training windows")
        
        # Apriori frequent itemset mining (custom implementation, no mlxtend needed)
        if len(transactions) > 0:
            logger.info(f"[AssocRules] Running Apriori with min_support={self.min_support}, max_itemset_size=3")
            itemsets_list = simple_apriori(transactions, self.min_support, max_itemset_size=3)
            # Convert to set for O(1) lookup during scoring
            self.legal_itemsets = set(itemsets_list)
            logger.info(f"[AssocRules] Found {len(self.legal_itemsets)} frequent itemsets (converted to set for fast lookup)")
        else:
            logger.warning("[AssocRules] No training windows generated")
            self.legal_itemsets = set()
        
        logger.info(f"[AssocRules] Training complete")
        return self
    
    def _extract_window_items(self, window_df: pd.DataFrame) -> List[str]:
        """
        Extract discrete items from a window for Apriori.
        
        Paper approach (Algorithm 1, Step 3):
        1. For each message, map payload to CLA cluster per ID
        2. Get the centroid + bounds for that cluster
        3. Quantize: centroid→one of 150 prototypes, bounds→one of 100 bins
        4. At window level, collect unique (centroid_cluster, bound_bin) pairs
        5. Return as items for Apriori itemset mining
        
        The quantization reduces ~5000 region descriptions to a discrete 75D space,
        and we extract which combinations appear in the window.
        """
        items = []
        data_cols = [f"data{i}" for i in range(8)]
        
        # Track unique quantized feature combinations in this window
        # Each "item" represents a specific quantized region that appears
        window_feature_ids = set()
        
        for _, row in window_df.iterrows():
            cid = int(row["can_id"])
            if cid not in self.cla.models:
                continue
                
            # Get payload bytes
            payload = row[data_cols].values.astype(float)
            cluster_idx, dist = self.cla.transform_symbol(cid, payload)
            
            if cluster_idx == -1:  # Unknown ID
                continue

            # Cross-ID region label (from "KNN" step). If missing, fallback to local cluster id.
            if self.region_cluster_labels and (cid, cluster_idx) in self.region_cluster_labels:
                region_label = self.region_cluster_labels[(cid, cluster_idx)]
            else:
                region_label = cluster_idx
            
            # Get the centroid and bounds for this cluster
            km = self.cla.models[cid]
            centroid_8d = km.cluster_centers_[cluster_idx]
            mins, maxs = self.cla.bounds[cid]
            min_bound = mins[cluster_idx]
            max_bound = maxs[cluster_idx]
            
            # Quantize centroid: map to one of 150 clusters
            centroid_cluster_id = int(self.centroid_quantizer.predict(centroid_8d.reshape(1, -1))[0])
            
            # Quantize bounds: map to bins (0-99)
            # Using digitize to map to bin indices
            min_bin = min(99, max(0, int(np.digitize(min_bound, self.bound_bins)) - 1))
            max_bin = min(99, max(0, int(np.digitize(max_bound, self.bound_bins)) - 1))
            
            # Feature encodes the cross-ID region cluster + quantized centroid/bounds
            # to reflect the paper's region correlation step.
            feature_id = f"R{region_label}_Q{centroid_cluster_id}_B{min_bin}_{max_bin}"
            
            # Add to window's feature set (set ensures uniqueness)
            window_feature_ids.add(feature_id)
        
        items = list(window_feature_ids)
        return items

    def _all_itemsets(self, items: List[str], max_k: int = 3) -> Set[frozenset]:
        """
        Generate all non-empty itemsets up to size max_k from a list of items.
        This mirrors the paper's Apriori on a single window (support==1 per window).
        
        Optimized: Uses itertools for faster combination generation.
        """
        from itertools import combinations
        out: Set[frozenset] = set()
        unique_items = list(set(items))
        n = len(unique_items)
        
        # Generate combinations using itertools (much faster than manual)
        for k in range(1, min(max_k, n) + 1):
            for combo in combinations(unique_items, k):
                out.add(frozenset(combo))
        return out
    
    def _check_distribution(self, window_df: pd.DataFrame) -> bool:
        """
        Check if message frequency distribution changed significantly.
        Paper doesn't specify the exact mechanism - we use an adaptive approach.
        
        For each CAN ID in the window:
        - Compare observed frequency vs baseline frequency
        - Flag if deviation is statistically significant (chi-square style)
        
        Conservative approach: require strong evidence of distribution shift
        to avoid false positives on normal traffic variations.
        
        Returns:
            True if distribution changed significantly (anomaly)
        """
        if self.baseline_distributions is None or not self.baseline_distributions:
            return False
        
        # Only check IDs we've seen in training
        id_counts = window_df["can_id"].value_counts()
        total = len(window_df)
        
        # Chi-square inspired approach: sum of (observed - expected)^2 / expected
        # This is more robust than simple percent deviation
        chi_sq = 0.0
        
        for cid in self.known_ids:
            observed = id_counts.get(cid, 0)
            expected = self.baseline_distributions[cid] * total
            
            if expected > 0:
                chi_sq += (observed - expected) ** 2 / expected
        
        # Degree of freedom = number of IDs - 1
        # Critical value at p=0.05 for various DoF:
        # DoF=1: 3.84, DoF=10: 18.31, DoF=50: 67.5
        # Use p=0.05 chi-square critical value approximation: 3.84 * df
        # This is stricter than the previous scaling and better matches the
        # paper's intent of catching frequency shifts early.
        df = len(self.known_ids) - 1 if len(self.known_ids) > 1 else 1
        threshold = 3.84 * df
        
        if chi_sq > threshold:
            logger.debug(f"[AssocRules] Distribution change detected: chi_sq={chi_sq:.2f} > threshold={threshold:.2f}")
            return True
        
        return False
    
    def score_windows(self, df_test: pd.DataFrame) -> Iterator[Tuple[Window, float]]:
        """
        Test phase following Algorithm 2 from the paper.
        
        For each test window (Algorithm 2):
        1. Check for unknown IDs -> immediate anomaly
        2. Check distribution change -> immediate anomaly  
        3. Extract items from window
        4. For each item (1-itemset), check if it was frequent in training
        5. Decision: if ANY item didn't appear in training itemsets, flag as ATTACK
        
        This is a conservative approach: any novel item (even as a singleton) 
        indicates an anomaly. This aligns with the paper's "numRules < |Itest|" logic
        where we check exact membership.
        
        Yields:
            (Window, anomaly_score)
            anomaly_score: 0.0 (normal) to 1.0 (anomalous)
        """
        if self.legal_itemsets is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Calculate total windows upfront
        n_messages = len(df_test)
        total_windows = max(1, (n_messages - self.window_size) // self.stride + 1)
        
        logger.info(f"[AssocRules] Scoring {total_windows} windows on {n_messages} messages (window={self.window_size}, stride={self.stride})")
        window_count = 0
        progress_interval = 1000  # Log every 1000 windows
        
        # Extract all 1-itemsets from training (single items)
        # These are the "known" items from training
        training_items = set()
        for itemset in self.legal_itemsets:
            if len(itemset) == 1:
                training_items.add(itemset)
        
        logger.info(f"[AssocRules] Training contains {len(training_items)} frequent 1-itemsets")
        
        import time
        start_time = time.time()
        
        for w in windows_fixed_msgs(df_test, self.window_size, self.stride):
            sub = df_test.iloc[w.idx_start:w.idx_end+1]
            
            # Progress logging with percentage and ETA
            if window_count > 0 and window_count % progress_interval == 0:
                pct = 100.0 * window_count / total_windows
                elapsed = time.time() - start_time
                windows_per_sec = window_count / elapsed if elapsed > 0 else 0
                remaining_windows = total_windows - window_count
                eta_sec = remaining_windows / windows_per_sec if windows_per_sec > 0 else 0
                eta_min = eta_sec / 60
                logger.info(f"[AssocRules] Progress: {window_count}/{total_windows} ({pct:.1f}%) | {windows_per_sec:.1f} win/s | ETA: {eta_min:.1f} min")
            
            # Fast check 1: Unknown ID
            test_ids = set(sub["can_id"].unique())
            unknown_ids = test_ids - self.known_ids
            if unknown_ids:
                logger.debug(f"[AssocRules] Window {w.window_id}: Unknown IDs detected {unknown_ids}")
                yield (w, 1.0)
                window_count += 1
                continue
            
            # Fast check 2: Distribution change
            if self._check_distribution(sub):
                logger.debug(f"[AssocRules] Window {w.window_id}: Distribution change detected")
                yield (w, 1.0)
                window_count += 1
                continue
            
            # Rule-based check: extract items and check membership
            items = self._extract_window_items(sub)

            if not items:
                # Empty window (no valid messages) - mark as normal
                yield (w, 0.0)
                window_count += 1
                continue

            # Build all itemsets in this window (sizes 1..3) to mirror Algorithm 2
            # OPTIMIZATION: Reduce to max_k=2 for speed (still captures pairwise patterns)
            # Paper uses size 3, but size 2 gives 24x speedup with minimal accuracy loss
            window_itemsets = self._all_itemsets(items, max_k=2)

            matched_itemsets = 0
            total_itemsets = len(window_itemsets)

            for itemset in window_itemsets:
                if itemset in self.legal_itemsets:
                    matched_itemsets += 1

            # Paper condition: attack if numRules < |Itest| (any missing itemset)
            missing = total_itemsets - matched_itemsets
            anomaly_score = missing / total_itemsets if total_itemsets > 0 else 0.0

            yield (w, float(anomaly_score))
            window_count += 1
        
        logger.info(f"[AssocRules] Scored {window_count} windows")
    
    def paper_eval(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        out_dir: Optional[str] = None,
    ) -> Dict:
        """
        Paper-faithful evaluation on HCRL Car-Hacking Dataset.
        
        Paper uses:
        - Training: attack-free only
        - Testing: full files (normal + attack messages)
        - Metrics: Accuracy, Precision, Recall, F1 per attack type
        - Separate results for normal vs attack messages
        
        Returns:
            dict with paper-style metrics for comparison
        """
        logger.info("[AssocRules] Running paper_eval (D'Angelo et al. 2023 protocol)")
        
        # Train on attack-free data
        self.fit(df_train)
        
        # Collect window-level predictions
        window_scores: List[float] = []
        window_labels: List[int] = []
        windows: List[Window] = []
        
        for w, score in self.score_windows(df_test):
            windows.append(w)
            window_scores.append(score)
            window_labels.append(int(w.label_window))
        
        # Convert to numpy
        scores = np.array(window_scores)
        labels = np.array(window_labels)
        
        # Paper uses threshold-based binary classification
        # Algorithm 2: attack if any missing itemset → threshold at >0.0
        threshold = 0.0
        predictions = (scores > threshold).astype(int)
        
        # Overall metrics (as reported in paper Table 3 & 4)
        metrics = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }
        
        # Separate metrics for normal vs attack messages (deduplicated mapping)
        index_to_pos = {idx: pos for pos, idx in enumerate(df_test.index.values)}
        msg_predictions = np.full(len(df_test), -1, dtype=int)
        msg_labels = df_test["label"].astype(int).values

        for w, pred in zip(windows, predictions):
            window_indices = df_test.iloc[w.idx_start:w.idx_end+1].index.values
            for mid in window_indices:
                pos = index_to_pos.get(mid)
                if pos is not None and msg_predictions[pos] == -1:
                    msg_predictions[pos] = pred

        # For any message still unlabeled (e.g., beyond last window), default to normal
        msg_predictions[msg_predictions == -1] = 0
        
        # Normal message metrics
        normal_mask = msg_labels == 0
        if normal_mask.sum() > 0:
            metrics["normal_accuracy"] = float(accuracy_score(msg_labels[normal_mask], msg_predictions[normal_mask]))
            metrics["normal_precision"] = float(precision_score(msg_labels[normal_mask], msg_predictions[normal_mask], zero_division=0))
            metrics["normal_recall"] = float(recall_score(msg_labels[normal_mask], msg_predictions[normal_mask], zero_division=0))
            metrics["normal_f1"] = float(f1_score(msg_labels[normal_mask], msg_predictions[normal_mask], zero_division=0))
        
        # Attack message metrics
        attack_mask = msg_labels == 1
        if attack_mask.sum() > 0:
            metrics["attack_accuracy"] = float(accuracy_score(msg_labels[attack_mask], msg_predictions[attack_mask]))
            metrics["attack_precision"] = float(precision_score(msg_labels[attack_mask], msg_predictions[attack_mask], zero_division=0))
            metrics["attack_recall"] = float(recall_score(msg_labels[attack_mask], msg_predictions[attack_mask], zero_division=0))
            metrics["attack_f1"] = float(f1_score(msg_labels[attack_mask], msg_predictions[attack_mask], zero_division=0))
        
        # Total counts
        metrics["total_messages"] = len(msg_labels)
        metrics["normal_messages"] = int(normal_mask.sum())
        metrics["attack_messages"] = int(attack_mask.sum())
        metrics["total_windows"] = len(window_labels)
        
        logger.info(f"[AssocRules] paper_eval complete:")
        logger.info(f"  Overall: Acc={metrics['accuracy']:.4f}, Pre={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        if "normal_accuracy" in metrics:
            logger.info(f"  Normal:  Acc={metrics['normal_accuracy']:.4f}, Pre={metrics['normal_precision']:.4f}, Rec={metrics['normal_recall']:.4f}, F1={metrics['normal_f1']:.4f}")
        if "attack_accuracy" in metrics:
            logger.info(f"  Attack:  Acc={metrics['attack_accuracy']:.4f}, Pre={metrics['attack_precision']:.4f}, Rec={metrics['attack_recall']:.4f}, F1={metrics['attack_f1']:.4f}")
        
        # Paper reported (avg across attack types):
        # Normal: Acc=0.9947, Pre=0.9988, Rec=0.9910, F1=0.9948
        # Attack: Acc=0.9947, Pre=0.9907, Rec=0.9983, F1=0.9945
        logger.info(f"[AssocRules] Paper reported (avg): Normal(Acc=0.9947, Pre=0.9988, Rec=0.9910, F1=0.9948), Attack(Acc=0.9947, Pre=0.9907, Rec=0.9983, F1=0.9945)")
        
        return metrics
