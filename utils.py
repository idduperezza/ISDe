from random import randint

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def load_mnist_data():

    """
    Loads data
    :return: x = data set, y = labels
    """

    data = read_csv(
        "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv")
    data = np.array(data)  # casting DataFrame to ndarray
    x = data[:, 1:]
    y = data[:, 0]
    return x, y


def plot_ten_digits(x, y=None):
    """
    Plots 10 images showing hand written digits
    :param x: data set
    :param y: labels
    """
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img = x[i, :].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        if y is not None:
            plt.title("Label: " + str(y[i]))


def save_plotted_images_in_pdf(images, labels, title):
    """
    Saves a plot into a PDF file
    :param images: array of unique images
    :param labels: unique labels, from 0 to 9
    :param title: title of the PDF file
    """
    plot_ten_digits(images, labels)
    plt.suptitle(title)
    plt.savefig('./images/' + title + 'images.pdf')


def randomize_ten_unique_images(x, y):
    """
    Randomizes 10 unique images taken from the dataset
    :param x: data set
    :param y: labels
    :return: unique images = randomized unique images, unique labels
    """
    unique_labels = np.unique(y)
    unique_images = np.zeros(shape=(unique_labels.size, x.shape[1]))
    for i in range(unique_labels.size):
        all_images_of_current_label = x[y == unique_labels[i], :]
        random_index = randint(0, all_images_of_current_label.shape[0])
        unique_images[i, :] = all_images_of_current_label[random_index, :]
    return unique_images, unique_labels

