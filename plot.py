from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def pretty_plot(x, y, z, title):
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
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Customize the axes.
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

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

	# Save output
	plt.savefig("../figures/" + str(title) + ".png")