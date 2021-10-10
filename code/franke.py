# for information about Franke's function see https://www.sfu.ca/~ssurjano/franke2d.html

import numpy as np

def function(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def noisy_function(x, y):
    franke = function(x, y)
    noise = np.random.normal(0,0.1,len(x)**2)
    noise = np.reshape(noise,(len(x),len(x)))
    return franke + noise