import math
import cvxpy as cp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8,9])

fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(a,b,c)
ax.set_title('Trajectory')

fig2, ax = plt.subplots()
ax.plot(a,[10,20,30])
fig2.suptitle('Position')

plt.show()