from conv_1d_kernels import CConvKernel


class CConvKernelCombo(CConvKernel):

    def kernel_mask(self):
        super().kernel_mask()

    def combo(self, x, *kernel_masks):
        """
        Applies multiple filters passed as input
        :param x: vector to modify
        :param kernel_masks: filters to apply to the vector
        :return: filtered vector
        """
        xp_prev = x.copy()

        for mask in kernel_masks:
            xp = self.kernel(xp_prev, mask)
            xp_prev = xp.copy()

        return xp