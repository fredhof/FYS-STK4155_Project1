# Requiers Python 3, numpy, matplotlib, pandas, scikit_learn and dependables
# note: x**2 is faster than x*x, but x*x*x is faster than x**3 etc.

def main():
    import numpy as np
    import plots, franke, methods, analysis
    from sklearn.model_selection import train_test_split
    from imageio import imread

    SEED = 8 # set seed for reproduceability

    
    """ Franke function data """

    # Make dataset
    N = int(1000)
    x, y, xx, yy, z = franke.prepare_data(N, noisy=True)

    # Plot data
    plots.plot_2d(z, title='Franke_data-2D')
    plots.plot_3d(xx, yy, z, title='Franke_data-3D')


    """OLS"""

    # Plot model
    plots.plot_model(x, y, xx, yy, z, regression_method='ols', max_degree=10, SEED=SEED, title='Franke-ols-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='ols', resampling=None, max_degree=10, SEED=SEED, title='Franke')


    """OLS with bootstrap"""


    """OLS with cross-validation"""


    """Ridge with boootstrap"""


    """Lasso with boootstrap"""


        # Benchmark
    """
    Compare with sklearn, and benchmark.
    """


    """ Terrain data """

    # Load 2D data
    z = imread('../data/SRTM_data_Norway_1.tif')
    z = z[:N, :N]

    # Make 3D data
    x = np.linspace(0,1, np.shape(z)[0])
    y = np.linspace(0,1, np.shape(z)[1])
    xx, yy = np.meshgrid(x, y)

    # Plot data
    plots.plot_2d(z, 'SRTM data in 2D')
    plots.plot_3d(xx, yy, z, 'SRTM data in 3D')


    """OLS"""

    # Plot model
    plots.plot_model(x, y, xx, yy, z, regression_method='ols', max_degree=10, SEED=SEED, title='SRTM-ols-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='ols', resampling=None, max_degree=10, SEED=SEED, title='SRTM')


if __name__ == '__main__':
    main()