from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class CConvKernel(ABC):

    def __init__(self, kernel_size=3):
        """
        Initializes the class
        :param kernel_size: mask dimension, it has to be an odd integer
        """
        if kernel_size % 2 == 0:
            raise TypeError("The kernel size must be an odd integer")

        self._kernel_size = kernel_size
        self._mask = None

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError("This method has yet to be implemented")

    def kernel(self, x, mask=None):
        """
        Function that applies a mask to an input vector x. If no mask is given as parameter the class mask is used by
        default
        :param x: vector to modify
        :param mask: filter to apply to the input vector
        :return: filtered vector/image
        """
        vector = deepcopy(x)
        half_dim = int(self.kernel_size / 2)

        if mask is None:
            mask = self._mask

        vector[half_dim:vector.size - half_dim] = \
            np.rint([np.dot(vector[(i - half_dim):(i + half_dim + 1)], mask)
                     for i in range(half_dim, vector.size - half_dim)]).astype(int)

        return vector

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, new_size):
        self._kernel_size = new_size
        self._mask = self.kernel_mask()

    @property
    def mask(self):
        return self._mask
