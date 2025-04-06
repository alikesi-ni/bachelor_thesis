import numpy as np


class DotProductKernel:
    """  Dot product kernel """

    @staticmethod
    def compute(*args) -> float:

        """
        Computes the dot product between feature vectors.

        Parameters
        ----------
        args : tuple of dictionary/ dictionaries
            Feature vector/ vectors

        Returns
        -------
        out : float
            Calculated dot product

        """

        if len(args) == 1:

            fv = np.fromiter(args[0].values(), float)
            return np.dot(fv, fv)

        elif len(args) == 2:

            common_keys = set(args[0].keys()).intersection(args[1].keys())

            val = 0

            for key in common_keys:
                val += args[0][key] * args[1][key]

            return val

        raise ValueError("Cannot compute dot product for more than 2 features!!!")
