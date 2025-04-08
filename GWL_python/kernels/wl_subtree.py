import numpy as np

from GWL_python.kernels.dot_product import DotProductKernel


class WLSubtreeKernel:
    """
    Weisfeiler Leman Subtree Kernel.

    Parameters
    ----------
    normalize : bool
        Whether to normalize the kernel matrix (default: False)

    """

    def __init__(self, normalize: bool = False):
        self.__normalize = normalize

    @staticmethod
    def compute(feature_vector: dict) -> float:

        """
        Computes the WLSubtreeKernel value of a single graph.

        Parameters
        ----------
        feature_vector : dict
            Feature vector of the graph

        Returns
        -------
        out : float
            Computed kernel value

        """

        return DotProductKernel.compute(feature_vector)

    def compute_kernel_matrix(self, feature_vectors: dict) -> np.array:

        """
        Computes the WLSubtreeKernel of all the given graphs.

        Parameters
        ----------
        feature_vectors : dict
            A dictionary, where key-value pair must be a graph id and corresponding feature vector

        Returns
        -------
        out : np.array
            Kernel matrix

        """

        fvs = list(feature_vectors.values())

        n = len(fvs)

        kernel_matrix = np.empty((n, n))
        kernel_matrix[:] = np.NaN

        for i in range(n):
            for j in range(n):

                if i == j:
                    kernel_matrix[i][j] = DotProductKernel.compute(fvs[i])
                else:
                    kernel_matrix[i][j] = DotProductKernel.compute(fvs[i], fvs[j])

        if self.__normalize:
            diagonal_elements = np.diagonal(kernel_matrix)
            kernel_matrix = np.divide(
                kernel_matrix, np.sqrt(np.outer(diagonal_elements, diagonal_elements))
            )
            np.nan_to_num(kernel_matrix, copy=False, nan=0, posinf=0, neginf=0)

        return kernel_matrix

    def fit_transform(self, feature_vectors: dict, train_mask: list[int]) -> np.array:

        """
        Computes the WLSubtreeKernel for all pairs of training graphs. Imitates Python's Scikit-Learn library's
        fit_transform(...) method.

        Parameters
        ----------
        feature_vectors : dict
            A dictionary, where key-value pair must be a graph id and corresponding feature vector

        train_mask : list[int]
            A list of graphs selected as training set

        Returns
        -------
        out : np.array
            A symmetric kernel matrix of shape (K_train, K_train)

        """

        n = len(train_mask)

        kernel_matrix = np.empty((n, n))
        kernel_matrix[:] = np.NaN

        for i, train_idx1 in enumerate(train_mask):
            for j, train_idx2 in enumerate(train_mask):
                kernel_matrix[i][j] = DotProductKernel.compute(
                    feature_vectors[train_idx1], feature_vectors[train_idx2])

        if self.__normalize:
            diagonal_elements = np.diagonal(kernel_matrix)
            kernel_matrix = np.divide(
                kernel_matrix, np.sqrt(np.outer(diagonal_elements, diagonal_elements))
            )
            np.nan_to_num(kernel_matrix, copy=False, nan=0, posinf=0, neginf=0)

        return kernel_matrix

    def transform(self, feature_vectors: dict, train_mask: list[int], test_mask: list[int]) -> np.array:

        """
        Computes the WLSubtreeKernel between pairs of test and training graphs. Imitates Python's Scikit-Learn library's
        transform(...) method.

        Parameters
        ----------
        feature_vectors : dict
            A dictionary, where key-value pair must be a graph id and corresponding feature vector

        train_mask : list[int]
            A list of graphs selected as training set

        test_mask : list[int]
            A list of graphs selected as test set

        Returns
        -------
        out : np.array
            A kernel matrix of shape (K_test, K_train)

        """

        kernel_matrix = np.empty((len(test_mask), len(train_mask)))
        kernel_matrix[:] = np.NaN

        for i, test_idx in enumerate(test_mask):
            for j, train_idx in enumerate(train_mask):
                kernel_matrix[i][j] = DotProductKernel.compute(
                    feature_vectors[test_idx], feature_vectors[train_idx])

        if self.__normalize:
            x_diagonal = [DotProductKernel.compute(feature_vectors[mask]) for mask in train_mask]
            y_diagonal = [DotProductKernel.compute(feature_vectors[mask]) for mask in test_mask]

            kernel_matrix = np.divide(
                kernel_matrix, np.sqrt(np.outer(y_diagonal, x_diagonal))
            )
            np.nan_to_num(kernel_matrix, copy=False, nan=0, posinf=0, neginf=0)

        return kernel_matrix
