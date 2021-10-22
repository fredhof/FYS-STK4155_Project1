# for information about Franke's function see https://www.sfu.ca/~ssurjano/franke2d.html
import numpy as np


def function(x, y):
    """
    Returns the Franke's function that has two Gaussian peaks of different heights and a smaller dip. 

    Args:
        x (np.Array[float]):        Inputs within [0, 1]
        y (np.Array[float]):        Inputs within [0, 1]

    Returns:
        z (np.Array[float]):        Outputs of the Franke's function

    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4


def noisy_function(x, y):
    """
    Returns the noisy Franke's function that has two Gaussian peaks of different heights and a smaller dip. 
    The noise is an added stochastic noise with the normal distribution N[0, 1].

    Args:
        x (np.Array[float]):        Inputs within [0, 1]
        y (np.Array[float]):        Inputs within [0, 1]

    Returns:
        z (np.Array[float]):        Outputs of the Franke's function with added noise

    """

    franke = function(x, y)
    noise = np.random.normal(0,0.1,len(x)**2)
    noise = np.reshape(noise,(len(x),len(x)))

    return franke + noise


def prepare_data(N, noisy):
    """

    """

    # Make dataset
    #x = np.arange(0, 1, 0.05)
    #y = np.arange(0, 1, 0.05)
    x = np.sort(np.random.uniform(0, 1, N)) # random uniform distribution with
    y = np.sort(np.random.uniform(0, 1, N)) # with x, y E [0, 1]
    xx, yy = np.meshgrid(x, y)

    if noisy==True:
        z = noisy_function(xx, yy)
    elif noisy==False:
        z = function(xx, yy)

    return x, y, xx, yy, z
