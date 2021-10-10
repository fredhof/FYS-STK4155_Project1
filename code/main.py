# Requiers Python 3, numpy, matplotlib, pandas, scikit_learn and dependables
# note: x**2 is faster than x*x, but x*x*x is faster than x**3 etc.

def main():

    import numpy as np
    import pandas as pd
    import sklearn as skl
    import franke, analysis, resampling, plot

    seed = 8
    np.random.seed(seed) # set seed for reproduceability

    #Exercise 1 - OLS on Franke function with train-test split

    # Make Data
    N = 1e3
    #x = np.arange(0, 1, 0.05)
    #y = np.arange(0, 1, 0.05)
    x = np.sort(np.random.uniform(0, 1, int(N))) # random uniform distribution with
    y = np.sort(np.random.uniform(0, 1, int(N))) # with x, y E [0, 1]
    xx, yy = np.meshgrid(x,y)
    z = franke.function(xx, yy)
    nz = franke.noisy_function(xx, yy)

    # Plot default
    plot.pretty_plot(x, y, z, "Franke function")
    plot.pretty_plot(x, y, nz, "Noisy Franke function")

    # Split data into test and train
    xy_combined = np.vstack((x, y)).T
    xy_train, xy_test, z_train, z_test = skl.model_selection.train_test_split((xy_combined), z, test_size=0.25, random_state=seed, shuffle=True)

    # Scaling data
    scaler = skl.preprocessing.StandardScaler().fit(xy_train)
    # call with scaled_train = scaler.transform(xy_train)

    analysis = analysis.analysis(x=xx,y=yy,lmd=0,degree=5,method='ols',design="yes")
    #MSE = analysis.MSE()
    #R2 = analysis.R2()
    #conf_interval = analysis.conf_interval()

    #Exercise 2 - Bias-variance trade-off and resampling (bootstrap)
    #s
    
    #Exercise 3 - Cross-validation resampling
    #s

    #Exercise 4 - Ridge Regression on the Franke function with resampling
    #s

    #Exercise 5 - Lasso Regression on the Franke function with resampling
    #s

    #Exercise 6 - Analysis of real data
    #s


if __name__ == '__main__':
    main()