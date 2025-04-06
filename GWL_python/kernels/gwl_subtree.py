from kernels.wl_subtree import WLSubtreeKernel


class GWLSubtreeKernel(WLSubtreeKernel):
    """
    Gradual Weisfeiler Leman Subtree Kernel. Basically, same as the Weisfeiler Leman Subtree Kernel, only difference
    is that the color refinement must be performed using Gradual Weisfeiler Leman algorithm.

    Parameters
    ----------
    normalize : bool
        Whether to normalize the kernel matrix (default: False)

    """

    def __init__(self, normalize: bool = False):
        super().__init__(normalize)
