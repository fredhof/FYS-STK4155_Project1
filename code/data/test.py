import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform

# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')

N = 1000
m = 5 # polynomial order
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z = terrain

def create_X(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    P = int((degree+1)*(degree+2)/2)
    X = np.ones((N,P))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = (x**(i-j))*(y**j)

    return X

#X = create_X(x_mesh, y_mesh,m)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
