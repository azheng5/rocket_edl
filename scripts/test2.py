#%% Modules
import math
import cvxpy as cp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#%% Initialize simulation

# Simulation config
N = 500 # number of time steps
dt = 0.1
time = np.linspace(0, N*dt, N+1)

# Rocket
m_fuel = 10000
m_dry = 25600
m_wet = m_fuel + m_dry

# Environment
g = 9.807
g_i = np.array([0,0,-g])
ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])

# Propulsion
Isp = 311
T_max = 411000
T_min = 0.0*T_max
alpha = 1/(Isp*g)

# Position control config
glideslope = 1*(np.pi/180)
thrust_cone = 30*(np.pi/180) # max thrust angle
v_max = 10000
v_horiz_max = 10000
rx0 = 0
ry0 = 0
rz0 = 100
vx0 = 0
vy0 = 0
vz0 = -10

#%% Variables
u = cp.Variable((N,3)) 
sigma = cp.Variable(N)
r = cp.Variable((N+1,3))
v = cp.Variable((N+1,3))
z = cp.Variable(N+1)

#%% Objective
objective = cp.sum(-z[N])

#%% Constraints
constraints = []

for k in range(N):
    # Dynamics constraints
    #TODO turn this constraint into vector form?
    rx_constraint = r[k+1,0] == r[k,0] + v[k,0]*dt + 0.5 * (dt**2) * (g_i[0] + u[k,0])
    ry_constraint = r[k+1,1] == r[k,1] + v[k,1]*dt + 0.5 * (dt**2) * (g_i[1] + u[k,1])
    rz_constraint = r[k+1,2] == r[k,2] + v[k,2]*dt + 0.5 * (dt**2) * (g_i[2] + u[k,2])
    vx_constraint = v[k+1,0] == v[k,0] + dt * (g_i[0] + u[k,0])
    vy_constraint = v[k+1,1] == v[k,1] + dt * (g_i[1] + u[k,1])
    vz_constraint = v[k+1,2] == v[k,2] + dt * (g_i[2] + u[k,2])

    mass_constraint = z[k+1] == z[k] - alpha*sigma[k]*dt

    # Control constraints
    thrust_norm_constraint = cp.norm(u[k,:]) <= sigma[k]
    thrust_cone_constraint = u[k,2] <= sigma[k] * math.cos(thrust_cone)
    sigma_lower_bound = sigma[k] >= mu1[k] * (1 - (z[k] - z0[k]) + 0.5*(z[k] - z0[k])**2)
    sigma_upper_bound = sigma[k] <= mu2[k] * (1 - (z[k] - z0[k]))
    z_lower_bound = z[k] >= math.log(m_wet - alpha*T_max*k*dt)
    z_upper_bound = z[k] <= math.log(m_wet - alpha*T_min*k*dt)

    thrust_lower_bound = z[k] >= T_min
    thrust_upper_bound = z[k] <= T_max
    # print(T[k,:] @ ez)

    # State constraints
    max_velocity = cp.norm(v[k,:]) <= v_max
    max_horiz_velocity_y = v[k,0] <= v_horiz_max
    max_horiz_velocity_z = v[k,1] <= v_horiz_max


    # Append constraints
    constraints.extend([rx_constraint, ry_constraint, rz_constraint, 
                        vx_constraint, vy_constraint, vz_constraint, mass_constraint, 
                        max_velocity, max_horiz_velocity_y, max_horiz_velocity_z,
                        thrust_lower_bound, thrust_upper_bound, thrust_norm_constraint, thrust_cone_constraint])

# Boundary value constraints
initial_mass = z[0] == math.log(m_wet)
final_mass = m[-1] >= math.log(m_dry)
initial_position_x = r[0,0] == rx0
initial_position_y = r[0,1] == ry0
initial_position_z = r[0,2] == rz0
initial_velocity_x = v[0,0] == vx0
initial_velocity_y = v[0,1] == vy0
initial_velocity_z = v[0,2] == vz0
final_position_x = r[-1,0] == 0
final_position_y = r[-1,1] == 0
final_position_z = r[-1,2] == 0
final_velocity_x = v[-1,0] == 0
final_velocity_y = v[-1,1] == 0
final_velocity_z = v[-1,2] == 0

constraints.extend([initial_mass, final_mass, initial_position_x, initial_position_y, initial_position_z,
                    initial_velocity_x, initial_velocity_y, initial_velocity_z, final_position_x, 
                    final_position_y, final_position_z, final_velocity_x, final_velocity_y, final_velocity_z])

#%% Solve
print("Solving problem...")
prob = cp.Problem(cp.Minimize(objective), constraints)

#%% Results
result = prob.solve()

fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(r.value[:,0],r.value[:,1],r.value[:,2])
ax.set_title('3d Trajectory')

fig2, ax = plt.subplots()
ax.plot(time,m.value[:])
fig2.suptitle('Mass')

plt.show()

#