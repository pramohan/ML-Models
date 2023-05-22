# Plotting Functions

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_losses(title, losses, save=False, ylabel="loss"):
    plt.clf()
    plt.scatter(np.arange(len(losses)), losses)
    plt.title(title)
    plt.xlabel ( "epochs" )
    plt.ylabel ( ylabel )
    if (save):
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/{title}.png')
    else:
        plt.show()

def plot_accuracies(title, accuracies, save=False):
    plot_losses(title, accuracies, save, ylabel="accuracies")

def plot_gradients (grads, ax):
    for i in range(len(grads[0])):
        ax.scatter(np.arange(len(grads)), [grad[i] for grad in grads])
    ax.set_xlabel ( "epochs" )
    ax.set_ylabel ( "gradient values" )
    return ax

def generate_plot(x, y, std, error_type, ax=None):
    #
    # Generate a plot with error bars
    #
    if error_type == 'train':
        color = 'green'
    else:
        color = 'red'

    if ax == None:
        plt.plot(x, y, label=f'{error_type} error')
        plt.fill_between(x, y - std, y + std, alpha=0.2, facecolor=color)
    else:
        ax.plot(x, y, label=f'{error_type} error')
        ax.fill_between(x, y - std, y + std, alpha=0.2, facecolor=color)


def plot_error(train_error, test_error, x, axis_label, ax=None):
    #
    # Plot the test and train error over 3 runs of the same experiment for specified
    # of training samples
    # Input:
    #   train_error: a 3xX matrix of training errors per run
    #   test_error: a 3xX matrix of test errors per run
    #   x: a 1xX vector that depicts the category that is varying (ie LR, epochs,
    #      sample size)
    #   axis_label: the xaxis label
    #   ax: default to None, otherwise plot on the axis passed in
    #
    # Across the 3 runs of the same experiment, calculate the mean error.
    # The output should be a 1xX vector
    #
    train_mean_errors = np.mean(train_error, axis=0)
    test_mean_errors = np.mean(test_error, axis=0)

    # Across the 3 runs of the same experiment, calculate the standard deviation.
    # The output should be a 1x5 vector
    train_std_devs = np.std(train_error, axis=0)
    test_std_devs = np.std(test_error, axis=0)

    # plt.clf() # clear previous plot

    # Axes labels
    ax.set_xlabel(f'{axis_label}')
    ax.set_ylabel(f'Error')

    # Create a plot with shading to indicate error bars
    # Plot the training error
    generate_plot(x, train_mean_errors, train_std_devs, 'train', ax)

    # Then plot the testing error on the same figure
    generate_plot(x, test_mean_errors, test_std_devs, 'test', ax)

    ax.legend()
    return ax


def plot_error_xlogscale(train_error, test_error, x, xlabel, ax=None):
    ax = plot_error(train_error, test_error, x, xlabel, ax)
    ax.set_xscale("log")
    return ax


def get_errors(model, X_train, y_train, X_test, y_test, lr, changing_var, change_type):
    train_err = np.zeros((3, len(changing_var)))
    test_err = np.zeros((3, len(changing_var)))

    # defaults if unchanged
    epochs = 1000
    activation_func = 'relu'
    loss_func = 'negative_log_likelihood'
    batch_size = 16

    for i in range(3):
        tr_errs = []
        test_errs = []
        for v in changing_var:
            if change_type == "epochs":
                model.train(X_train, y_train, v, lr, batch_size, activation_func, loss_func)
            elif change_type == "learning_rate":
                model.train(X_train, y_train, epochs, v, batch_size, activation_func, loss_func)
            elif change_type == "batch_size":
                model.train(X_train, y_train, epochs, lr, v, activation_func, loss_func)
            elif change_type == 'activation_func':
                model.train(X_train, y_train, epochs, lr, batch_size, v, loss_func)
            elif change_type == 'loss_func':
                model.train(X_train, y_train, epochs, lr, batch_size, activation_func, v)
            tr_errs.append(model.get_losses()[-1])
            model.predict(X_test,y_test)
            test_errs.append(model.test_loss)
        train_err[i, :] = tr_errs
        test_err[i, :] = test_errs
    return train_err, test_err, model

def plot_training_differences (Ns, gaus_train_errors, gaus_test_errors, circle_train_erorrs, circle_test_errors, save=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))
    ax1 = plot_error_xlogscale ( gaus_train_errors.T, gaus_test_errors.T, Ns, 'N', ax1 )
    ax2 = plot_error_xlogscale ( circle_train_erorrs.T, circle_test_errors.T, Ns, 'N', ax2 )
    if save == True:
        plt.savefig('figs/training_size_differences.png')
    else:
        plt.show()


def plot_error_bar(x, y, std, error_type, ax=None):
    if error_type == 'train':
        color = 'green'
    else:
        color = 'red'

    if ax == None:
        plt.scatter(x, y, label=f'{error_type} error')
        plt.fill_between(x, y - std, y + std, alpha=0.2, facecolor=color)
    else:
        ax.scatter(x, y, label=f'{error_type} error')
        ax.fill_between(x, y - std, y + std, alpha=0.2, facecolor=color)


def plot_layer_size_error(train_error, test_error, layer_type, ax ):
    train_mean_errors = np.mean(train_error, axis=0)
    test_mean_errors = np.mean(test_error, axis=0)

    # Across the 3 runs of the same experiment, calculate the standard deviation.
    # The output should be a 1x5 vector
    train_std_devs = np.std(train_error, axis=0)
    test_std_devs = np.std(test_error, axis=0)

    # plt.clf() # clear previous plot

    # Axes labels
    ax.set_xlabel(f'Layers')
    ax.set_ylabel(f'Error')

    # Create a plot with shading to indicate error bars
    # Plot the training error
    plot_error_bar([layer_type], train_mean_errors, train_std_devs, 'train', ax)

    # Then plot the testing error on the same figure
    plot_error_bar([layer_type], test_mean_errors, test_std_devs, 'test', ax)

    ax.legend()
    return ax
