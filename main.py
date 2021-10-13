import numpy as np
from utils import load_mnist_data, save_plotted_images_in_pdf, randomize_ten_unique_images
from conv_1d_kernels import CConvKernelMovingAverage, CConvKernelTriangle, CConvKernelCombo

# initializing kernels
combo_kernel = CConvKernelCombo()
moving_average_kernel = CConvKernelMovingAverage()
triangle_kernel = CConvKernelTriangle()

# loading data
x, y = load_mnist_data()

# randomizing unique images per class
images, labels = randomize_ten_unique_images(x, y)

# plotting original images into a PDF
save_plotted_images_in_pdf(images, labels, "Without filter")

# applying moving average kernel
moving_average_images = np.array([moving_average_kernel.kernel(images[i])
                                  for i in range(labels.shape[0])])

# plotting moving average kernel images into a PDF
save_plotted_images_in_pdf(moving_average_images, labels, "Moving average filter")

# applying triangle kernel
triangle_images = np.array([triangle_kernel.kernel(images[i])
                            for i in range(labels.shape[0])])

# plotting triangle kernel images into a PDF
save_plotted_images_in_pdf(triangle_images, labels, "Triangle filter")

# applying combo kernel
combo_images = \
    np.array(
        [combo_kernel.combo(images[i], moving_average_kernel.mask, triangle_kernel.mask, moving_average_kernel.mask)
         for i in range(labels.shape[0])])

# plotting combo kernel images into a PDF
save_plotted_images_in_pdf(triangle_images, labels, "Combo filter")
