import numpy as np


# TODO: Consider Loss Class
def negative_log_likelihood(predicted, actual):
    samples = len(actual)
    correct_logprobs = -np.log(predicted[range(samples), actual])
    data_loss = np.sum(correct_logprobs) / samples
    return data_loss


def nll_derivative(predicted, actual):
    num_samples = len(actual)
    ## compute the gradient on predictions
    dscores = predicted
    dscores[range(num_samples), actual] -= 1
    dscores /= num_samples
    return dscores
