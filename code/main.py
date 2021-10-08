# Requiers Python 3, numpy, matplotlib, pandas, scikit_learn and dependables
# note: x**2 is faster than x*x, but x*x*x is faster than x**3 etc.

def main():

    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import pandas as pd
    import sklearn as skl
    import franke, analysis, resampling

    np.random.seed(8) # set seed for reproduceability

    #Exercice 1 - OLS on Franke function with train-test split
    N = 1e3
    x = np.random.rand(int(N))
    y = np.random.rand(int(N))
    xx, yy = np.meshgrid(x,y)
    z = franke.function(xx, yy)

    analysis = analysis.analysis(x=xx,y=yy,lmd=0,degree=5,method='ols',design="yes")
    #MSE = analysis.MSE()
    #R2 = analysis.R2()
    #conf_interval = analysis.conf_interval()

    #Exercice 2 - Bias-variance trade-off and resampling (bootstrap)
    s
    
    #Exercice 3 - Cross-validation resampling
    s

    #Exercice 4 - Ridge Regression on the Franke function with resampling
    s

    #Exercice 5 - Lasso Regression on the Franke function with resampling
    s

    #Exercice 6 - Analysis of real data
    s


if __name__ == '__main__':
    main()