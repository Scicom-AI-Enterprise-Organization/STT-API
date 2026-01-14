"""
PyTorch implementation of StreamingKMeansMaxCluster.

Keeps all operations on GPU tensors to avoid numpy conversions.
"""

import torch
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
