import math
import cvxpy as cp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any

from scripts.translational_control import solve_fixed_tof

dt = 0.1
m_fuel = 10000
m_dry = 25600
m_wet = m_fuel + m_dry
g = 9.807
g_i = np.array([0,0,-g])
Isp = 311
T_max = 411000
T_min = 0.4*T_max
alpha = 1/(Isp*g)
glideslope = 1*(np.pi/180)
thrust_cone = 10*(np.pi/180)
r0 = np.array([200,100,2000])
v0 = np.array([2,-3,-50])
v_max = 1000000 # arb
v_horiz_max = 1000000 # arb

config = {
    "m_fuel": m_fuel,
    "m_dry": m_dry,
    "m_wet": m_wet,
    "g": g,
    "g_i": g_i,
    "ex": np.array([1,0,0]),
    "ey": np.array([0,1,0]),
    "ez": np.array([0,0,1]),
    "Isp": Isp,
    "T_max": T_max,
    "T_min": T_min,
    "alpha": alpha,
    "glideslope": glideslope,
    "thrust_cone": thrust_cone,
    "v_max": v_max,
    "v_horiz_max":v_horiz_max,
    "dt":dt
}



## Time of flight
t_min = (m_dry*LA.norm([v0[0], v0[1], v0[2]]))/T_max
t_max = (m_wet - m_dry)/(alpha*T_min)
t_i = t_min

# Conduct time of flight search by finding the minimum feasible TOF
print("Conducting time of flight search..")
while t_i < t_max:
    print("Current TOF:", t_i)
    (soln, valid_tof) = solve_fixed_tof(r0,v0,t_i,m_wet,config)
    # m_final = soln["m"][-1]

    # Check if current TOF has a valid solution
    if valid_tof:# and m_final < m_dry + m_fuel:
        print("Optimal TOF found")
        t_star = t_i
        break
    else:
        t_i += 1



# #%% Plot
# fig1 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(log.r[:,0],log.r[:,1],log.r[:,2])
# ax.scatter(log.r[0,0],log.r[0,1],log.r[0,2],s=20)
# ax.scatter(log.r[-1,0],log.r[-1,1],log.r[-1,2], marker='x',color='blue')
# ax.scatter(0,0,0,marker='x',color='red',s=30)
# ax.set_title('3d Trajectory')

# #%%
# fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
# ax1.plot(time,log.r[:,0])
# ax2.plot(time,log.r[:,1])
# ax3.plot(time,log.r[:,2])
# fig2.suptitle('Position')

# #%%
# fig3, (ax1,ax2,ax3) = plt.subplots(3,1)
# ax1.plot(time,log.v[:,0])
# ax2.plot(time,log.v[:,1])
# ax3.plot(time,log.v[:,2])
# fig3.suptitle('Velocity')

# #%%
# fig4, (ax1,ax2,ax3) = plt.subplots(3,1)
# ax1.plot(time[0:-1],log.T[:,0])
# ax2.plot(time[0:-1],log.T[:,1])
# ax3.plot(time[0:-1],log.T[:,2])
# fig4.suptitle('Thrust')

# #%%
# fig5, ax = plt.subplots()
# ax.plot(time,log.m)
# fig5.suptitle('Mass')

# #%%
# fig6, ax = plt.subplots()
# ax.plot(time[0:-1],LA.norm(log.T,axis=1))
# fig6.suptitle('Thrust Norm')

# #%%
# plt.show()

print('fin.')
