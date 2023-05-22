import os
import sklearn
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random_control import *


kGaussian = 'Gaus'
kCircles = 'Circles'
kIris = 'Iris'

# This class is provided. Nothing should need to be done in it unless you would like to tweak it 
# yourself
class Dataset():    
    def __init__(self, dist0, dist1):
        if dist0 is None or dist1 is None:
            return
        self.N = dist0.distribution().shape[0]
        self.M = dist1.distribution().shape[0]
        dims = dist0.distribution().shape[1]
        dataset_points = np.zeros((self.N + self.M, dims + 1))
        dataset_points[0:self.N, -1] = np.ones((1, self.N))  # set N points to 1 (M points to 0)
        dataset_points[0:self.N, :-1] = dist0.distribution()
        dataset_points[self.N:, :-1] = dist1.distribution()

        self.dataset = dataset_points

        if dist0.get_type() == dist1.get_type():
            self.dataset_type = dist0.get_type()
        else:
            self.dataset_type = "mixed"

    def get_dataset(self):
        return self.dataset

    def save_dataset_plot(self):
        os.makedirs('figs', exist_ok=True)
        self.show_dataset(save=True)
        print('Dataset plot has been saved in figs/')

    def show_dataset(self, save=False):
        #
        # Helper function to visualize the dataset generated in 'generate_nd_dataset'
        # This is not possible with a dataset generated from generate_nd_dataset
        # if dims > 2
        #
        if self.dataset_type == kIris:
            print("Plotting for Iris is not enabled")
        else:
            assert self.dataset.shape[1] == 3, "Dataset is not 2-dimensional. Abort."

            y = self.dataset[:, -1]
            x_one_class = self.dataset[(self.dataset[:, -1] == 1)]
            x_zero_class = self.dataset[(self.dataset[:, -1] == 0)]

            plt.plot(x_one_class[:, 0], x_one_class[:, 1], 'x')
            plt.plot(x_zero_class[:, 0], x_zero_class[:, 1], 'o')

            plt.title("Visualization of Synthetically Generated Data")

            plt.axis('equal')
            if (save):
                plt.savefig('figs/{}.png'.format(self.dataset_type))
            else:
                plt.show()

class IrisDataset(Dataset):
    def __init__(self, dataset_points):
        super().__init__(None,None)
        self.dataset = dataset_points
        self.dataset_type = kIris

class Distribution():
    def __init__(self, n, dims):
        self.n = n
        self.dims = dims
        self.dataset_points = np.zeros((n, dims))

    def distribution(self):
        return self.dataset_points

    def get_type(self):
        return "None"


class GaussianDistribution(Distribution):
    def __init__(self, n, dims):
        global generator 
        super().__init__(n, dims)

        mu = generator.randint(50, size=(dims))  # Generate a random number between 0 and 50
        d = generator.randint(100, size=(1, dims))[0]
        cov = np.diag(d)

        X = generator.multivariate_normal(mu, cov, n)
        self.dataset_points = X

    def get_type(self):
        return kGaussian


class CircleDistribution(Distribution):
    def __init__(self, n, f):
        super().__init__(n, 2)
        noise = 0.1

        linspace = np.linspace(0, 2 * np.pi, self.n)
        x1 = np.cos(linspace) * f
        x2 = np.sin(linspace) * f

        X = np.vstack(
            [x1, x2]
        ).T

        # add some noise
        X += generator.normal(scale=noise, size=X.shape)
        self.dataset_points = X

    def get_type(self):
        return kCircles


class IrisDistribution(Distribution):
    def __init__(self):
        X, y = load_iris(return_X_y=True)
        super().__init__(X.shape[0], X.shape[1])
        n_points, dims = X.shape
        self.dataset_points = np.zeros((n_points, dims+1))
        self.dataset_points[:,:-1] = X
        self.dataset_points[:,-1] = y

    def get_type(self):
        return kIris


## Class-less function
def generate_nd_dataset(N, M, distribution, dims=2):
    # Input:
    #   N: a number of points in the dataset belonging to one class
    #   M: a number of points in the dataset belonging to another class
    #   dims: The dimensionality of the dataset. Default to 2.
    #
    # Each row represents a point, each column an x,y,and 1 or 0 indicating the gaussian split
    #
    if distribution == kGaussian:
        # gaussians can be n-dimensional
        gaus0 = GaussianDistribution(N, dims)
        gaus1 = GaussianDistribution(M, dims)
        dataset = Dataset(gaus0, gaus1)

    elif distribution == kCircles:
        # circles is only 2d
        circle0 = CircleDistribution(N, 2.0)
        circle1 = CircleDistribution(N, 0.5)
        dataset = Dataset(circle0, circle1)

    elif distribution == kIris:
        # Iris is only 3d
        iris = IrisDistribution()
        dataset = IrisDataset(iris.dataset_points)

    return dataset


def train_test_split_ ( dataset ):
    #
    # Input: the dataset output from from generate_nd_dataset
    #
    # Splits the dataset into 80% train, 20% test
    #
    X = dataset[:,:-1]
    y = dataset[:,-1]
    # TODO: Part 4 A: hint scikit-learn's train_test_split function can be used.

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = seed)
    # return X_train, X_test, np.expand_dims(y_train, axis=1), np.expand_dims(y_test, axis=1)
    return X_train, X_test, y_train.astype(int), y_test.astype(int)