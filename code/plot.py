from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np

import franke, analysis, methods

def pretty_plot(x_mesh, y_mesh, z, title):
	"""
	Plots a 3d surface coloured with the coolwarm colourmap.

	Inputs:
	x - array of x floats with len(n)
	y - array of y floats with len(n)
	z - array of z floats with len(n)
	title - string containing the title of the graph/name of the output file

	Outputs:
	None

	Saves produced figure to ../figures/title.png.
	"""
		
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	# Plot the surface.
	surf = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Customize the axes.
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

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
	
	# Save output, .pdf for vector graphics
	plt.savefig("../article/figures/" + str(title) + ".pdf")

def analysis_plot(x, y, z, reg_method, lmd, max_degree, specific_plot = False):

	nz = z + np.random.normal(0,0.1,len(x))

	xaxis = np.linspace(1,max_degree,max_degree)
	MSE = np.zeros(max_degree) 
	R2 = np.zeros(max_degree)

	for i in range(max_degree):
		X = methods.design_matrix(x,y,degree=i)
		reg = methods.regression(X,nz,reg_method,lmd=lmd)
		z_pred = reg.predict()

		analyze = analysis.analysis(X, z, z_pred)
		MSE[i] =  analyze.MSE()
		R2[i] = analyze.R2()

	# for all figures
	title = r"$\lambda = {}$".format(lmd)
	plt.title(title)
	plt.xlabel('Polynomial degree')

	plt.figure(1)
	plt.plot(xaxis, MSE, label=reg_method)
	plt.ylabel('MSE')
	plt.figure(2)
	plt.plot(xaxis, R2, label=reg_method)
	plt.ylabel('R2')
	

	if specific_plot == 'MSE':
		plt.close(2) # closes figure 2
	elif specific_plot == 'R2':
		plt.close(1)

	return #returns figures
