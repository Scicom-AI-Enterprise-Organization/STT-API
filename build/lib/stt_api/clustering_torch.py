"""
PyTorch implementations of online clustering algorithms for speaker diarization.

- StreamingKMeansMaxClusterTorch: online K-means with max cluster cap
- StreamingBIRCHTorch: online BIRCH with cosine distance

All operations stay on GPU tensors to avoid numpy conversions.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


class StreamingKMeansMaxClusterTorch:
    """
    Streaming KMeans with maximum cluster size - PyTorch implementation.

    Parameters
    ----------
    threshold: float
        Minimum threshold to consider new cluster.
    max_clusters: int
        Maximum cluster size.
    """

    def __init__(self, threshold: float, max_clusters: int = 5):
        self.max_clusters = max_clusters
        self.threshold = threshold
        self.cluster_centers: List[torch.Tensor] = []
        self.cluster_sizes: List[int] = []
        self.labels: List[int] = []

    def fit(self, data: torch.Tensor):
        """Fit on batch of data."""
        for sample in data:
            self.streaming(sample)

    def streaming(self, sample) -> int:
        """
        Process a single sample and return its cluster label.

        Parameters
        ----------
        sample: torch.Tensor
            Single embedding vector

        Returns
        -------
        int: Cluster label for the sample
        """
        if len(self.cluster_centers) == 0:
            self.cluster_centers.append(sample.clone())
            self.cluster_sizes.append(1)
            self.labels.append(0)
            return 0

        # Calculate distances to all existing centers
        centers_tensor = torch.stack(
            self.cluster_centers
        )  # [num_centers, embedding_dim]
        sample_expanded = sample.unsqueeze(0)  # [1, embedding_dim]

        # Cosine distance (1 - cosine_similarity)
        cosine_sim = torch.nn.functional.cosine_similarity(
            centers_tensor, sample_expanded, dim=1
        )
        distances = 1.0 - cosine_sim

        # Find nearest cluster
        nearest_cluster = torch.argmin(distances).item()
        min_distance = distances[nearest_cluster].item()

        if min_distance <= self.threshold:
            # Update existing cluster
            old_center = self.cluster_centers[nearest_cluster]
            old_size = self.cluster_sizes[nearest_cluster]
            new_size = old_size + 1

            # Update center: (old_center * old_size + sample) / new_size
            self.cluster_centers[nearest_cluster] = (
                old_center * old_size + sample
            ) / new_size
            self.cluster_sizes[nearest_cluster] = new_size

        elif len(self.cluster_centers) < self.max_clusters:
            # Create new cluster
            self.cluster_centers.append(sample.clone())
            self.cluster_sizes.append(1)
            nearest_cluster = len(self.cluster_centers) - 1

        else:
            # Replace farthest cluster with new sample
            distances_to_centers = torch.norm(centers_tensor - sample_expanded, dim=1)
            farthest_cluster = torch.argmax(distances_to_centers).item()

            if self.cluster_sizes[farthest_cluster] > 1:
                # Reduce size and update center
                old_center = self.cluster_centers[farthest_cluster]
                old_size = self.cluster_sizes[farthest_cluster]
                new_size = old_size - 1

                # Remove the sample's contribution (approximate by removing average)
                self.cluster_centers[farthest_cluster] = (
                    old_center * old_size - sample
                ) / new_size
                self.cluster_sizes[farthest_cluster] = new_size
            else:
                # Replace cluster entirely
                self.cluster_centers[farthest_cluster] = sample.clone()
                self.cluster_sizes[farthest_cluster] = 1

            nearest_cluster = farthest_cluster

        self.labels.append(nearest_cluster)
        return nearest_cluster


class StreamingBIRCHTorch:
    """
    Streaming BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
    - PyTorch implementation using cosine distance.

    Each subcluster stores a Clustering Feature (CF): (n, linear_sum).
    The centroid is derived as linear_sum / n and compared via cosine similarity.

    Key difference from StreamingKMeansMaxClusterTorch:
        When at max_clusters capacity a new point triggers a merge of the two
        closest subclusters (by cosine distance between centroids) before the
        new point is added as its own subcluster.  This preserves more cluster
        history than K-means' "replace farthest" strategy.

    Parameters
    ----------
    threshold : float
        Cosine distance threshold below which a sample is absorbed into the
        nearest subcluster (same scale as StreamingKMeansMaxClusterTorch).
    max_clusters : int
        Maximum number of live subclusters at any time.
    """

    def __init__(self, threshold: float, max_clusters: int = 5):
        self.threshold = threshold
        self.max_clusters = max_clusters
        # CF linear sums (unnormalized); centroid = cluster_sum[i] / cluster_n[i]
        self.cluster_sum: List[torch.Tensor] = []
        self.cluster_n: List[int] = []
        self.labels: List[int] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _centroid(self, idx: int) -> torch.Tensor:
        """Return the (unnormalized) centroid of subcluster idx."""
        return self.cluster_sum[idx] / self.cluster_n[idx]

    def _centroids_matrix(self) -> torch.Tensor:
        """Stack all centroids into a [K, D] tensor."""
        return torch.stack([self._centroid(i) for i in range(len(self.cluster_n))])

    def _merge_two_closest(self) -> int:
        """
        Find the pair of subclusters with the highest cosine similarity
        (lowest cosine distance), merge the second into the first, and
        return the surviving cluster index.

        Uses a vectorised similarity matrix for efficiency.
        """
        k = len(self.cluster_n)
        centroids = self._centroids_matrix()                        # [K, D]
        norms = F.normalize(centroids, dim=1)                       # [K, D]
        sim_matrix = torch.mm(norms, norms.t())                     # [K, K]

        # Mask the diagonal so self-pairs are never chosen
        sim_matrix.fill_diagonal_(-2.0)

        flat_idx = torch.argmax(sim_matrix).item()
        merge_i = int(flat_idx) // k
        merge_j = int(flat_idx) % k

        # Keep merge_i < merge_j so removal of merge_j is safe
        if merge_i > merge_j:
            merge_i, merge_j = merge_j, merge_i

        # Merge CF: accumulate linear sum and count
        self.cluster_sum[merge_i] = self.cluster_sum[merge_i] + self.cluster_sum[merge_j]
        self.cluster_n[merge_i] += self.cluster_n[merge_j]

        # Remove the absorbed subcluster
        self.cluster_sum.pop(merge_j)
        self.cluster_n.pop(merge_j)

        # Remap historical labels
        for k_idx in range(len(self.labels)):
            lbl = self.labels[k_idx]
            if lbl == merge_j:
                self.labels[k_idx] = merge_i
            elif lbl > merge_j:
                self.labels[k_idx] = lbl - 1

        return merge_i

    # ------------------------------------------------------------------
    # Public API (mirrors StreamingKMeansMaxClusterTorch)
    # ------------------------------------------------------------------

    def streaming(self, sample: torch.Tensor) -> int:
        """
        Process a single embedding and return its subcluster label.

        Parameters
        ----------
        sample : torch.Tensor
            Single speaker embedding vector (any device).

        Returns
        -------
        int : Subcluster label for this sample.
        """
        if not self.cluster_n:
            self.cluster_sum.append(sample.clone())
            self.cluster_n.append(1)
            self.labels.append(0)
            return 0

        # Cosine distance from sample to every subcluster centroid
        centroids = self._centroids_matrix()                        # [K, D]
        cosine_sim = F.cosine_similarity(centroids, sample.unsqueeze(0), dim=1)
        distances = 1.0 - cosine_sim                                # [K]

        nearest = int(torch.argmin(distances).item())
        min_dist = distances[nearest].item()

        if min_dist <= self.threshold:
            # Absorb into nearest subcluster (update CF linear sum)
            self.cluster_sum[nearest] = self.cluster_sum[nearest] + sample
            self.cluster_n[nearest] += 1
            self.labels.append(nearest)
            return nearest

        if len(self.cluster_n) < self.max_clusters:
            # Room for a new subcluster
            new_idx = len(self.cluster_n)
            self.cluster_sum.append(sample.clone())
            self.cluster_n.append(1)
            self.labels.append(new_idx)
            return new_idx

        # At capacity: merge the two closest subclusters, then add new one
        self._merge_two_closest()
        new_idx = len(self.cluster_n)
        self.cluster_sum.append(sample.clone())
        self.cluster_n.append(1)
        self.labels.append(new_idx)
        return new_idx

    def fit(self, data: torch.Tensor):
        """Fit on a batch of embeddings."""
        for sample in data:
            self.streaming(sample)
