from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from imageio import imread

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

	Saves produced figure to ../figures/title.png.

	"""
	
	plt.figure()	
	ax = plt.subplots(subplot_kw={"projection": "3d"})

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

	# Add a color bar which maps values to colors.
	plt.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
	
	# Save output
	plt.savefig("../article/figures/" + str(title) + ".png")


def plot_2d(data, title):
	"""
	Plots a 3d surface coloured with the coolwarm colourmap.

	Args:
	data (np.Array['H, W', float]):		Data matrix
	title (string):						Title of the graph/output file

	Returns:
		None

	Saves produced figure to ../figures/title.png.

	"""

	plt.figure()

	# Plot the surface
	show = plt.imshow(data, cmap=cm.coolwarm)

	# Customize
	plt.title(title)
	plt.colorbar(show)

	# Save output
	plt.savefig("../article/figures/" + str(title) + ".png")