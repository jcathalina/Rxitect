from typing import Callable

from torch import nn


def mlp(num_in: int, num_hid: int, num_out: int, num_layers: int, activation_fn: Callable = nn.LeakyReLU) -> nn.Sequential:
    """Creates a fully-connected network with no activation after the last layer.
    If `num_layers` is 0 then this corresponds to `nn.Linear(num_in, num_out)`.

    Parameters
    ----------
    num_in
        Size of input
    num_out
        Size of output
    num_hid
        Size of hidden layers
    num_layers
        The number of layers to build, if set to 0, returns `nn.Linear(num_in, num_out)`
    activation_fn
        The activation function to use at each hidden layer
    """
    n = [num_in] + [num_hid] * num_layers + [num_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), activation_fn()] for i in range(num_layers + 1)], [])[:-1])
