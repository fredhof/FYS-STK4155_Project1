# Requiers Python 3, numpy, matplotlib, pandas, scikit_learn and dependables
# note: x**2 is faster than x*x, but x*x*x is faster than x**3 etc.

def main():

	import numpy as np
	import pandas as pd
	import sklearn as skl
	import franke, analysis, methods, plot
	from imageio import imread
	import matplotlib.pyplot as plt

	seed = 8
	np.random.seed(seed) # set seed for reproduceability

	#Exercise 1 - OLS on Franke function with train-test split

	# Make Data
	N = int(1e3)
	#x = np.arange(0, 1, 0.05)
	#y = np.arange(0, 1, 0.05)
	x = np.sort(np.random.uniform(0, 1, N)) # random uniform distribution with
	y = np.sort(np.random.uniform(0, 1, N)) # with x, y E [0, 1]

	xx, yy = np.meshgrid(x,y)
	z = franke.function(xx, yy)
	nz = franke.noisy_function(xx, yy)

	# Plot default
	#plot.pretty_plot(xx, yy, z, "Franke function")
	#plot.pretty_plot(xx, yy, nz, "Noisy Franke function")

	# Split data into test and train
	xy_combined = np.vstack((x, y)).T
	xy_train, xy_test, z_train, z_test = skl.model_selection.train_test_split((xy_combined), z, test_size=0.25, random_state=seed, shuffle=True)

	# Scaling data
	scaler = skl.preprocessing.StandardScaler().fit(xy_train)
	# call with scaled_train = scaler.transform(xy_train)


	# Resampling
	# methods.resampling( input values )  

	# regression example
	X = methods.design_matrix(x,y,degree=5)
	reg = methods.regression(X,nz,"ridge",lmd=1e-3)
	z_pred = reg.predict()

	reg2 = methods.regression(X,nz,'skl_ridge',lmd=1e-3)
	z_pred2 = reg2.predict()
	#plot.pretty_plot(xx, yy, z_pred,'lasso')
	
	#boots = methods.resampling(X,nz,'bootstrap')



	analysis = analysis.analysis(X, z, z_pred)
	analysis.run_comparison_tests(X, z, z_pred2, error = 1e-12)

	#Exercise 2 - Bias-variance trade-off and resampling (bootstrap)
	#s
	
	#Exercise 3 - Cross-validation resampling
	#s

	#Exercise 4 - Ridge Regression on the Franke function with resampling
	#s

	#Exercise 5 - Lasso Regression on the Franke function with resampling
	#s

	#Exercise 6 - Analysis of real data
	
	terrain = imread('../code/data/SRTM_data_Norway_1.tif')
	terrain = terrain[:N,:N]
	# Creates mesh of image pixels
	x = np.linspace(0,1, np.shape(terrain)[0])
	y = np.linspace(0,1, np.shape(terrain)[1])
	x_mesh, y_mesh = np.meshgrid(x,y)
	X2 = methods.design_matrix(x,y,degree=5)
	reg3 = methods.regression(X2,terrain,'ridge',lmd=1e-3)
	z_pred3 = reg3.predict()

	#plot.pretty_plot(x_mesh,y_mesh,z_pred3,'ridge terrain')

	# doesn't work if 3d initialized
	# Compares MSE over different degrees
	plot.analysis_plot(x, y, z, 'ridge', lmd=1e-3, max_degree=16)
	plot.analysis_plot(x, y, z, 'ols', lmd=1e-3, max_degree=16)
	#plot.analysis_plot(x, y, z, 'skl_lasso', lmd=1e-3, max_degree=16) # takes a really long time...
	
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()