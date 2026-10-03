import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 200, endpoint=False)

x = np.cos(theta)
y = np.sin(theta)

A = np.array([[2, 1],
              [1, 3]])

points = np.column_stack((x, y))
new_points = points @ A.T

eigenvalues, eigenvectors = np.linalg.eigh(A)

plt.scatter(x, y, s=5)
plt.scatter(new_points[:,0], new_points[:,1], s=5)

plt.quiver(0, 0,
           eigenvectors[0,0], eigenvectors[1,0],
           angles='xy', scale_units='xy', scale=1)

plt.quiver(0, 0,
           eigenvectors[0,1], eigenvectors[1,1],
           angles='xy', scale_units='xy', scale=1)

plt.axis('equal')
plt.grid()
plt.show()

## also not doing the positive definite related codes for now.