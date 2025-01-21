import math as m

import cvxpy as cp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any
plt.style.use('dark_background')

import rkt_config

def tof_search(r0, v0, m0, tof_guess, config):

    # Extract variables
    dt_c = config["dt_c"]
    m_dry = config["m_dry"]
    T_min = config["T_min"]
    T_max = config["T_max"]
    alpha = config["alpha"]

    ## Time of flight
    t_min = (m_dry*LA.norm([v0[0], v0[1], v0[2]]))/T_max
    t_max = (m0 - m_dry)/(alpha*T_min)
    t_i = t_min

    # Initialize TOF guess
    if tof_guess <= t_max:
        t_i = tof_guess

    # Conduct time of flight search by finding the minimum feasible TOF
    print("Searching for optimal TOF: ", end='')

    while t_i < t_max:
        print(t_i,' ',end='')
        (soln, valid_tof) = solve_fixed_tof(r0,v0,t_i,m0,config)
        # m_final = soln["m"][-1]

        # Check if current TOF has a valid solution
        if valid_tof: # and m_final < m_dry + m_fuel:
            print("")
            print("Optimal TOF found")

            # next iterations guess should be a slightly lower value
            tof_guess = t_i - dt_c
            # if t_i <= 15:
            #     tof_guess = 1
            # else:
            #     tof_guess = t_i - 3*dt_c

            return soln, tof_guess
        else: # modify tof and continue searching for valid tof
            t_i -= 0.2*dt_c
            # if t_i < 15:
            #     t_i += dt_c
            # elif t_i <= 8:
            #     t_i += 0.1*dt_c
            # else:
            #     t_i += dt_c

def solve_fixed_tof(r0, v0, tof, m0, config):

    # Extract variables
    dt_c = config["dt_c"]
    N = int(tof/dt_c)
    time = np.linspace(0, N*dt_c, N+1)
    g = config["g"]
    g_i = config["g_i"]
    ex = config["ex"]
    ey = config["ey"]
    ez = config["ez"]
    T_max = config["T_max"]
    T_min = config["T_min"]
    alpha = config["alpha"]
    glideslope = config["glideslope"]
    thrust_cone = config["thrust_cone"]
    v_max = config["v_max"]
    v_horiz_max = config["v_horiz_max"]
    m_dry = config['m_dry']

    # Variables
    u = cp.Variable((N,3))
    sigma = cp.Variable(N)
    r = cp.Variable((N+1,3))
    v = cp.Variable((N+1,3))
    z = cp.Variable(N+1)

    # Objective
    objective = cp.sum(-z[N])

    # Constraints
    constraints = []

    for k in range(N):
        # Dynamics constraints
        #TODO turn this constraint into vector form?
        rx_dynamics = r[k+1,0] == r[k,0] + v[k,0]*dt_c + 0.5 * (dt_c**2) * (g_i[0] + u[k,0])
        ry_dynamics = r[k+1,1] == r[k,1] + v[k,1]*dt_c + 0.5 * (dt_c**2) * (g_i[1] + u[k,1])
        rz_dynamics = r[k+1,2] == r[k,2] + v[k,2]*dt_c + 0.5 * (dt_c**2) * (g_i[2] + u[k,2])
        vx_dynamics = v[k+1,0] == v[k,0] + dt_c * (g_i[0] + u[k,0])
        vy_dynamics = v[k+1,1] == v[k,1] + dt_c * (g_i[1] + u[k,1])
        vz_dynamics = v[k+1,2] == v[k,2] + dt_c * (g_i[2] + u[k,2])

        z_dynamics = z[k+1] == z[k] - alpha*sigma[k]*dt_c

        # Control constraints
        z0 =  m.log(m0 - alpha*T_max*k*dt_c)
        mu1 = T_min*m.exp(-z0)
        mu2 = T_max*m.exp(-z0)
        thrust_norm_constraint = cp.norm(u[k,:]) <= sigma[k]
        thrust_cone_constraint = u[k,0] <= sigma[k] * m.cos(thrust_cone)
        sigma_lower_bound = sigma[k] >= mu1 * (1 - (z[k] - z0) + 0.5*(z[k] - z0)**2)
        sigma_upper_bound = sigma[k] <= mu2 * (1 - (z[k] - z0))
        z_lower_bound = z[k] >= m.log(m0 - alpha*T_max*k*dt_c)
        z_upper_bound = z[k] <= m.log(m0 - alpha*T_min*k*dt_c)

        # State constraints
        # glideslope_constraint = cp.norm(cp.vstack([r[k,0], r[k,1]])) - r[k,2] * m.tan(np.pi/2 - glideslope) <= 0
        glideslope_constraint = r[k,0] >= 0
        max_velocity = cp.norm(v[k,:]) <= v_max
        max_horiz_velocity_y = v[k,1] <= v_horiz_max
        max_horiz_velocity_z = v[k,2] <= v_horiz_max

        # Append constraints
        constraints.extend([rx_dynamics, ry_dynamics, rz_dynamics, 
                            vx_dynamics, vy_dynamics, vz_dynamics, z_dynamics,
                            glideslope_constraint,
                            max_velocity, max_horiz_velocity_y, max_horiz_velocity_z,
                            thrust_norm_constraint, thrust_cone_constraint,
                            sigma_lower_bound, sigma_upper_bound, z_lower_bound, z_upper_bound])

    # Boundary value constraints
    initial_position_x = r[0,0] == r0[0]
    initial_position_y = r[0,1] == r0[1]
    initial_position_z = r[0,2] == r0[2]
    initial_velocity_x = v[0,0] == v0[0]
    initial_velocity_y = v[0,1] == v0[1]
    initial_velocity_z = v[0,2] == v0[2]
    final_position_x = r[-1,0] == 0
    final_position_y = r[-1,1] == 0
    final_position_z = r[-1,2] == 0
    final_velocity_x = v[-1,0] == 0
    final_velocity_y = v[-1,1] == 0
    final_velocity_z = v[-1,2] == 0
    initial_mass = z[0] == m.log(m0)
    final_mass = z[-1] >= m.log(m_dry)

    constraints.extend([initial_position_x, initial_position_y, initial_position_z,
                        initial_velocity_x, initial_velocity_y, initial_velocity_z, final_position_x, 
                        final_position_y, final_position_z, final_velocity_x, final_velocity_y, final_velocity_z,
                        initial_mass, final_mass])

    # Solve
    prob = cp.Problem(cp.Minimize(objective), constraints)

    result = prob.solve(verbose=False)

    # Check if optimal solution found
    if prob.status not in ["infeasible","unbounded"]:
        m_temp = np.exp(z.value) # N
        z_temp = z.value.reshape(N+1,1)
        soln = {
            't': time,
            'r': r.value,
            'v': v.value,
            'm': m_temp,
            'T': np.exp(z_temp[0:-1,:]) * u.value,
            'u': u.value, # debug purposes
            'z': z.value
        }
        valid_tof = True
    else:
        valid_tof = False
        soln = result

    return (soln, valid_tof)

if __name__ == "__main__":
    '''Unit test - "offline" traj solver'''

    config = rkt_config.config

    (soln, tof_guess) = tof_search(config['r0'],config['v0'],config['m0'],config['init_tof_guess'],config)

    #%% Plot
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(soln['r'][:,2],-soln['r'][:,1],soln['r'][:,0])
    ax.scatter(soln['r'][0,2],-soln['r'][0,1],soln['r'][0,0],s=20)
    ax.scatter(soln['r'][-1,2],-soln['r'][-1,1],soln['r'][-1,0], marker='x',color='blue')
    ax.scatter(0,0,0,marker='x',color='red',s=30)
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_zlabel('x')
    ax.set_title('3d Trajectory')

    #%%
    fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(soln['t'],soln['r'][:,0])
    ax2.plot(soln['t'],soln['r'][:,1])
    ax3.plot(soln['t'],soln['r'][:,2])
    fig2.suptitle('Position')

    #%%
    fig3, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(soln['t'],soln['v'][:,0])
    ax2.plot(soln['t'],soln['v'][:,1])
    ax3.plot(soln['t'],soln['v'][:,2])
    fig3.suptitle('Velocity')

    #%%
    fig4, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(soln['t'][0:-1],soln['T'][:,0])
    ax2.plot(soln['t'][0:-1],soln['T'][:,1])
    ax3.plot(soln['t'][0:-1],soln['T'][:,2])
    fig4.suptitle('Thrust')

    #%%
    fig5, ax = plt.subplots()
    ax.plot(soln['t'],soln['m'])
    fig5.suptitle('Mass')

    #%%
    fig6, ax = plt.subplots()
    ax.plot(soln['t'][0:-1],LA.norm(soln['T'],axis=1))
    fig6.suptitle('Thrust Norm')

    #%%
    plt.show()

    print('fin.')