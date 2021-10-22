from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn import metrics
import methods, franke
import numpy as np
from sklearn.model_selection import train_test_split


def plot_3d(x, y, z, title):
	"""
	Plots a 3d surface coloured with the coolwarm colourmap.

	Args:
        x (np.Array[float]):        x-data
        y (np.Array[float]):        y-data
        z (np.Array[float]):        z-data
        title (string):				Title of the graph/output file

	Returns:
		None

	Saves produced figure to ../figures/<title>.pdf.

	"""
	
	#plt.figure()	
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	# Plot the surface
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Customize
	"""
	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('X', fontweight="bold")
	ax.set_ylabel('Y', fontweight="bold")
	ax.set_zlabel('Z', fontweight="bold")
	ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = -3, -2, -2 
	ax.set_title(title, fontweight="bold", y=1.0, pad=-15)
	ax.tick_params(axis='both', which='major', labelsize=8, pad=-1)
	ax.tick_params(axis='z', which='major', labelsize=8, pad=0.1)
	ax.view_init(15, -30)
	"""
	plt.title(title)
	plt.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
	
	# Save output
	plt.savefig("../article/figures/" + str(title) + ".pdf")


def plot_2d(data, title):
	"""
	Plots a 3d surface coloured with the coolwarm colourmap.

	Args:
		data (np.Array['H, W', float]):		Data matrix
		title (string):						Title of the graph/output file

	Returns:
		None

	Saves produced figure to ../figures/<title>.pdf.

	"""

	plt.figure()

	# Plot the surface
	show = plt.imshow(data, cmap=cm.coolwarm)

	# Customize
	plt.title(title)
	plt.colorbar(show)

	# Save output
	plt.savefig("../article/figures/" + str(title) + ".pdf")


def plot_model(x, y, xx, yy, z, regression_method, lambda_, max_degree, SEED, title):
	"""
	"""

	# Get data matrix
	X = methods.design_matrix(x, y, degree=max_degree)
	
	# Train
	model = methods.regression(X, z, regression_method=regression_method, lambda_=lambda_)

    # Get models
	z_pred_train = model.get_prediction(X)

    # Plot
	plot_3d(xx, yy, z_pred_train, title)


def plot_analytics(x, y, z, regression_method, resampling, lambda_, max_degree, SEED, title):
	"""
	"""

	# Initialize variables
	degrees = np.arange(1, max_degree)
	MSE_train = np.zeros(len(degrees))
	MSE_test = np.zeros(len(degrees))
	bias = np.zeros(len(degrees))
	variance = np.zeros(len(degrees))
	R2_score = np.zeros(len(degrees))

	for degree in range(max_degree-1):

		# Get data matrix
		X = methods.design_matrix(x, y, degree=degree)

		if resampling==None:

			# Split dataset
			X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=SEED, shuffle=True)

        	# Train
			ols = methods.regression(X_train, z_train, regression_method=regression_method, lambda_=lambda_)

        	# Get models
			z_pred_train = ols.get_prediction(X_train)
			z_pred_test = ols.get_prediction(X_test)

        	# Get Scores
			MSE_train[degree] = metrics.mean_squared_error(z_pred_train, z_train)
			MSE_test[degree] = metrics.mean_squared_error(z_pred_test, z_test)
			bias[degree] = np.mean((z_test - np.mean(z_pred_test))**2)
			variance[degree] = np.mean(np.var(z_pred_test, axis=1))
			R2_score[degree] = metrics.r2_score(z_test, z_pred_test)

		elif resampling=='Bootstrap':
			return

		elif resampling=='Cross-validation':
			return


	# Print best results:
	s_idx = np.where(MSE_test==MSE_test.min())
	print('Following configuration gives the smallest error:')
	print('MSE = {}, degree= {}, R2_score {}'.format(MSE_test[s_idx], degrees[s_idx], R2_score[s_idx]))


	# Create plots
	plot_MSE_test_train(degrees, MSE_train, MSE_test, '{}-{}-{}-MSE_test_train'.format(title, regression_method, resampling))
	plot_MSE_decomposition(degrees, MSE_test, bias, variance, '{}-{}-{}-MSE_decomposition'.format(title, regression_method, resampling))
	plot_R2_score(degrees, R2_score, '{}-{}-{}-R2_score'.format(title, regression_method, resampling))
	

def plot_MSE_test_train(degrees, MSE_train, MSE_test, title):
	"""Plot comparison of mean squared error of test and train set"""

	plt.figure()

	# Plot the graph
	plt.plot(degrees, MSE_train, label='MSE of train dataset')
	plt.plot(degrees, MSE_test, label='MSE of test dataset')

	# Customize
	plt.xlabel('Polynomal model complexity in degrees')
	plt.ylabel('Mean Squared Error')
	plt.legend()

	# Save output
	plt.savefig("../article/figures/" + str(title) + ".pdf")

	plt.close()


def plot_R2_score(degrees, R2_score, title):
	"""Plot coefficient of determination score R2"""

	plt.figure()

	# Plot the graph
	plt.plot(degrees, R2_score, label='R2 score of test dataset')

	# Customize
	plt.xlabel('Model complexity in degrees')
	plt.ylabel('Coefficient of determination score R2')
	plt.legend()

	# Save output
	plt.savefig("../article/figures/" + str(title)+ ".pdf")

	plt.close()


def plot_MSE_decomposition(degrees, MSE_test, bias, variance, title):
	"""Test mean squared error decomposition"""

	plt.figure()

	# Plot the graph
	plt.plot(degrees, MSE_test, label='MSE of test dataset')
	plt.plot(degrees, bias, label='bias of test dataset')
	plt.plot(degrees, variance, label='variance of test dataset')

	# Customize
	plt.xlabel('Model complexity in degrees')
	plt.ylabel('Mean Squared Error')
	plt.legend()

	# Save output
	plt.savefig("../article/figures/" + str(title) + ".pdf")

	plt.close()


def plot_confidence_interval():
	"""
	TO DO
	"""