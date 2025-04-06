from typing import Union

import numpy as np


class KMeans:
    """
    Custom implementation of KMeans clustering algorithm for GWL.

    Parameters
    ----------
    k : int
        Number of clusters

    initialization_method : str
        Cluster center initialization method. Must be either "forgy" or "kmeans++" (default: "kmeans++")

    max_iter : int
        Maximum number of iterations to be performed (default: 100)

    num_forgy_iterations : int
        Number of clustering iterations to be performed for selecting the best clustering using "forgy" initialization.
        Has effect only if the initialization method is "forgy" (default: 15)

    seed : int
        Seed for Numpy random functions. Ensures reproducibility of random selections/ choices (default: 1234)

    verbose : bool
        If True, prints every step of fitting and cluster assignment. Useful for debugging (default: False)

    """

    def __init__(self, k: int, initialization_method: str = "kmeans++", max_iter: int = 100,
                 num_forgy_iterations: int = 15, seed: int = 1234, verbose: bool = False) -> None:

        # Number of clusters
        self.k = k

        # Initialization method
        if initialization_method == "forgy" or initialization_method == "kmeans++":
            self.initialization_method = initialization_method
        else:
            raise ValueError("Unsupported initialization method !!! Supported methods are \"forgy\" and \"kmeans++\"")

        self.max_iter = max_iter
        self.num_forgy_iterations = num_forgy_iterations  # has effect only when the initialization method is forgy
        self.seed = seed
        self.verbose = verbose

    def fit_and_assign_cluster(self, neighbor_color_count: dict[int, list[int]]) -> dict:

        """
        Initializes k cluster centers as per defined initialization_method, fits the clusters, and updates
        their centers until convergence, and returns the clustered nodes.

        Convergence criteria: when all cluster membership remain unchanged or maximum number of iteration
        has been reached.

        Parameters
        ----------
        neighbor_color_count : dict[int, list[int]]
            A dictionary where key is an integer node id and value is a list containing the neighbor color count.

        Returns
        -------
        out : dict
            A dictionary where key is the cluster designation and corresponding value contain the nodes assigned to
            that cluster

        """

        nodes, data = list(zip(*neighbor_color_count.items()))
        data = np.array(data)

        n_samples, n_features = data.shape

        if self.verbose:
            print("\nInitializing cluster centers...")

        # number of cluster centers
        unique_data = np.unique(data, axis=0)
        n_unique_data = len(unique_data)

        # number of clusters
        n_clusters = min(self.k, n_unique_data)

        if n_clusters == 1:
            return {0: nodes}

        cluster_centers: Union[np.array, list[np.array]] = self.__initialize_cluster_centers(n_clusters, data)

        if self.verbose:
            print("\nCluster centers initialized...")

        if self.verbose:
            print("\nStarting cluster assignment and cluster center updation...")

        if self.initialization_method == "forgy":

            cluster_memberships_eta = dict()

            for i in range(self.num_forgy_iterations):

                cluster_centers_mu = cluster_centers[i]

                cluster_memberships = np.empty(n_samples, dtype=int)

                iteration = 1

                while True:

                    new_cluster_memberships = np.empty(n_samples, dtype=int)

                    # Assign
                    for idx, point in enumerate(data):
                        new_cluster_memberships[idx] = np.argmin(
                            KMeans.__calculate_squared_euclidean_distances(point, cluster_centers_mu))

                    # Update
                    for cluster in np.unique(new_cluster_memberships):
                        cluster_centers_mu[cluster] = np.mean(data[new_cluster_memberships == cluster], axis=0)

                    # Convergence criteria: when all cluster membership remain unchanged
                    if np.all(cluster_memberships == new_cluster_memberships):

                        if self.verbose:
                            print(f"\nForgy Clustering: {i}, Cluster centers: {cluster_centers_mu}")
                            print("Converged....")
                            print("Iteration required: ", iteration)

                        break

                    # or maximum number of iteration has been reached
                    elif iteration > self.max_iter:
                        if self.verbose:
                            print(f"\nForgy Clustering: {i}, Cluster centers: {cluster_centers_mu}")
                            print("Maximum iteration has been reached!")

                        break

                    else:
                        cluster_memberships = new_cluster_memberships.copy()

                    iteration += 1

                cluster_memberships_eta[i] = {
                    "memberships": cluster_memberships,
                    "wcss_score": KMeans.__calculate_wcss_score(data, cluster_centers_mu, cluster_memberships)
                }

                if self.verbose:
                    print(f"\nWCSS score for forgy Clustering {i}: {cluster_memberships_eta[i]['wcss_score']}")

            if self.verbose:
                print("\nCluster assignment and cluster center updation process finished.")
                print("\nSelecting best clustering using WCSS scores...")

            min_wcss_item = min(cluster_memberships_eta.items(), key=lambda item: item[1]["wcss_score"])
            cluster_memberships = min_wcss_item[1]["memberships"]

            if self.verbose:
                print("\nBest clustering selected.")

        else:

            cluster_memberships = np.empty(n_samples, dtype=int)

            iteration = 1

            while True:

                new_cluster_memberships = np.empty(n_samples, dtype=int)

                # Assign
                for idx, point in enumerate(data):
                    new_cluster_memberships[idx] = np.argmin(
                        KMeans.__calculate_squared_euclidean_distances(point, cluster_centers))

                # Update
                for cluster in np.unique(new_cluster_memberships):
                    cluster_centers[cluster] = np.mean(data[new_cluster_memberships == cluster], axis=0)

                # Convergence criteria: when all cluster membership remain unchanged
                if np.all(cluster_memberships == new_cluster_memberships):

                    if self.verbose:
                        print("\nConverged....")
                        print("Iteration required: ", iteration)

                    break

                # or maximum number of iteration has been reached
                elif iteration > self.max_iter:
                    if self.verbose:
                        print("Maximum iteration has been reached!")

                    break

                else:
                    cluster_memberships = new_cluster_memberships.copy()

                iteration += 1

            if self.verbose:
                print("\nCluster assignment and cluster center updation process finished.")

        if self.verbose:
            print("\nAssigning nodes to corresponding cluster...")

        # assigning nodes to corresponding cluster
        clusters = {cluster: [] for cluster in np.unique(cluster_memberships)}

        for node, cluster in zip(nodes, cluster_memberships):
            clusters[cluster] += [node]

        return clusters

    def __initialize_cluster_centers(self, n_clusters: int, data: np.array) -> Union[np.array, list[np.array]]:

        """
        Initializes the cluster centers.

        Parameters
        ----------
        n_clusters : int
            Number of cluster centers to initialize

        data : np.array
            A numpy array containing the neighbor color count vectors for clustering

        Returns
        -------
        out : Union[np.array, list[np.array]]
            Returns a list of numpy arrays of length num_forgy_iterations, where each array contains k cluster centers
            if the initialization method is forgy. Otherwise, returns a numpy array containing k cluster centers for
            kmeans++

        """

        rng = np.random.default_rng(self.seed)

        unique_data = np.unique(data, axis=0)
        n_unique_data = len(unique_data)

        if self.initialization_method == "forgy":

            return [unique_data[rng.choice(np.arange(n_unique_data), size=n_clusters, replace=False)] for _ in
                    range(self.num_forgy_iterations)]

        elif self.initialization_method == "kmeans++":

            cluster_centers = list()

            available_for_selection = np.ones(n_unique_data, dtype=bool)
            mu_zero_idx = rng.integers(n_unique_data)

            cluster_centers.append(unique_data[mu_zero_idx])
            available_for_selection[mu_zero_idx] = False

            available_data = unique_data[available_for_selection]

            for cluster_idx in range(1, n_clusters):

                distances_per_cluster_center = np.empty((len(cluster_centers), available_data.shape[0]))

                for idx, cluster_center in enumerate(cluster_centers):
                    distances = KMeans.__calculate_squared_euclidean_distances(cluster_center, available_data)
                    distances_per_cluster_center[idx] = distances

                min_distances = np.min(np.array(distances_per_cluster_center), axis=0)

                probabilities = min_distances / np.sum(min_distances)

                x_hat_idx = rng.choice(np.arange(available_data.shape[0]), 1, replace=False, p=probabilities)
                cluster_centers.append(available_data[x_hat_idx].reshape(-1))

                available_for_selection = np.ones(available_data.shape[0], dtype=bool)
                available_for_selection[x_hat_idx] = False

                available_data = available_data[available_for_selection]

            return np.array(cluster_centers)

        else:
            raise ValueError("Unsupported initialization method provided !!!")

    @staticmethod
    def __calculate_squared_euclidean_distances(vector: np.array, matrix: np.array) -> np.array:

        """
        Calculates the squared Euclidean distances between a given vector and each row of the matrix.
        Since, squaring cancels out the square-root, these two operations are not performed.

        Parameters
        ----------
        vector : np.array
            A numpy array of shape (m, 1)

        matrix : np.array
            A numpy array of shape (n, m)

        Returns
        -------
        out : np.array
            Squared Euclidean distances between a vector and the matrix rows

        """

        return np.sum(np.square(matrix - vector), axis=1)

    @staticmethod
    def __calculate_wcss_score(data: np.array, cluster_centers: np.array, cluster_memberships: np.array) -> float:

        """
        Calculates Intra-Cluster/ Within-Cluster Squared Sum (WCSS) score.

        Parameters
        ----------
        data : np.array
            A numpy array containing the neighbor color count vectors for clustering

        cluster_centers : np.array
            A numpy array containing the cluster centers

        cluster_memberships : np.array
            A numpy array containing the cluster membership of each vector in parameter data

        Returns
        -------
        out : float
            Calculated within cluster squared sum score

        """

        wcss_score = np.float64(0)

        for cluster in np.unique(cluster_memberships):
            distances = KMeans.__calculate_squared_euclidean_distances(
                cluster_centers[cluster], data[cluster_memberships == cluster]
            )
            wcss_score += np.sum(np.square(distances))

        return wcss_score
