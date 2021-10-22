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
    plots.plot_model(x, y, xx, yy, z, regression_method='ols', lambda_=None, max_degree=10, SEED=SEED, title='Franke-ols-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='ols', resampling=None, lambda_=None, max_degree=10, SEED=SEED, title='Franke')


    """OLS with bootstrap"""


    """OLS with cross-validation"""


    """Ridge"""

    # Plot model
    plots.plot_model(x, y, xx, yy, z, regression_method='ridge', lambda_=1e-1, max_degree=10, SEED=SEED, title='Franke-ridge-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='ridge', resampling=None, lambda_=1e-1, max_degree=10, SEED=SEED, title='Franke')

    
    """Ridge with boootstrap"""

    
    """Lasso"""

    
    # Plot model
    plots.plot_model(x, y, xx, yy, z, regression_method='sklearn_lasso', lambda_=1e-1, max_degree=10, SEED=SEED, title='Franke-lasso-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='sklearn_lasso', resampling=None, lambda_=1e-1, max_degree=10, SEED=SEED, title='Franke')
    

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
    plots.plot_model(x, y, xx, yy, z, regression_method='ols', lambda_=None, max_degree=10, SEED=SEED, title='SRTM-ols-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='ols', resampling=None, lambda_=None, max_degree=10, SEED=SEED, title='SRTM')


    """OLS with bootstrap"""


    """OLS with cross-validation"""


    """Ridge"""

    # Plot model
    plots.plot_model(x, y, xx, yy, z, regression_method='ridge', lambda_=1e-1, max_degree=10, SEED=SEED, title='SRTM-ridge-None-model')

    # Plot analytics
    plots.plot_analytics(x, y, z, regression_method='ridge', resampling=None, lambda_=1e-1, max_degree=10, SEED=SEED, title='SRTM')

    
    """Ridge with boootstrap"""

    
    """Lasso"""


    """Lasso with boootstrap"""




if __name__ == '__main__':
    main()