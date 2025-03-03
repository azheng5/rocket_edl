# Trying to debug MPC - conducting error analysis for different integrators over a
# trajectory with a fixed optimal thrust profile

import sys
import os
import copy

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from Rocket import Rocket
import sims.trajectory_control as trajectory_control
import rkt_config

if __name__ == '__main__':

    # Generate optimized trajectory
    config = rkt_config.config
    (soln0, tof_guess) = trajectory_control.tof_search(config['r0_mean'],config['v0_mean'],config['m0'],config['init_tof_guess'],config)

    # Initialize rocket integrators
    rk4 = Rocket(config)
    eul = Rocket(config)
    discrete = Rocket(config)
    ivp = Rocket(config)

    rk4.x = [config['r0_mean'][0], config['r0_mean'][1], config['r0_mean'][2],
             config['v0_mean'][0], config['v0_mean'][1], config['v0_mean'][2],
             config['m0']]
    eul.x = [config['r0_mean'][0], config['r0_mean'][1], config['r0_mean'][2],
             config['v0_mean'][0], config['v0_mean'][1], config['v0_mean'][2],
             config['m0']]
    discrete.x = [config['r0_mean'][0], config['r0_mean'][1], config['r0_mean'][2],
             config['v0_mean'][0], config['v0_mean'][1], config['v0_mean'][2],
             config['m0']]
    ivp.x = [config['r0_mean'][0], config['r0_mean'][1], config['r0_mean'][2],
             config['v0_mean'][0], config['v0_mean'][1], config['v0_mean'][2],
             config['m0']]

    rk4_log = copy.deepcopy(rk4.empty_log)
    eul_log = copy.deepcopy(eul.empty_log)
    discrete_log = copy.deepcopy(discrete.empty_log)
    ivp_log = copy.deepcopy(ivp.empty_log)



    # Integrate rockets
    k = 0
    t = 0
    while k < soln0['T'].shape[0]:

        T_des = soln0['T'][k,:]

        rk4_log['t'][k] = t
        rk4_log['r'][k,:] = rk4.x[0:3]
        rk4_log['v'][k,:] = rk4.x[3:6]
        rk4_log['m'][k] = rk4.x[6]
        rk4_log['T'][k,:] = T_des

        eul_log['t'][k] = t
        eul_log['r'][k,:] = eul.x[0:3]
        eul_log['v'][k,:] = eul.x[3:6]
        eul_log['m'][k] = eul.x[6]
        eul_log['T'][k,:] = T_des

        discrete_log['t'][k] = t
        discrete_log['r'][k,:] = discrete.x[0:3]
        discrete_log['v'][k,:] = discrete.x[3:6]
        discrete_log['m'][k] = discrete.x[6]
        discrete_log['T'][k,:] = T_des
    
        for i in range( int(config["dt_c"]/config["dt_sim"]) ):
            rk4.x = rk4.rk4_step(rk4.eom,t,rk4.x,T_des)
            eul.x = eul.euler_step(eul.eom,t,eul.x,T_des)
            discrete.x = discrete.discrete_eom(t,discrete.x,T_des)

            t += rk4.dt_sim

        k += 1

    t_span = (0,soln0['t'][-1])
    ivp_result = solve_ivp(lambda t, x: ivp.eom(t,x,soln0['T'],soln0['t']), t_span, ivp.x, 
                           t_eval= np.arange(t_span[0], t_span[1], config['dt_sim']),method='RK45')

    np.set_printoptions(precision=6,suppress=True)
    print('Soln:',soln0['r'][k-1,0:3])
    print('RK4:',rk4_log['r'][k-1,0:3])
    print('Euler:',eul_log['r'][k-1,0:3])
    print('Discrete:',discrete_log['r'][k-1,0:3])
    print('IVP:',ivp_result['y'][0:3,-1])

    # Plot
    fig1, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(rk4_log['t'][0:k],rk4_log['r'][0:k,0],label='RK4')
    ax2.plot(rk4_log['t'][0:k],rk4_log['r'][0:k,1],label='RK4')
    ax3.plot(rk4_log['t'][0:k],rk4_log['r'][0:k,2],label='RK4')
    ax1.plot(eul_log['t'][0:k],eul_log['r'][0:k,0],label='Euler')
    ax2.plot(eul_log['t'][0:k],eul_log['r'][0:k,1],label='Euler')
    ax3.plot(eul_log['t'][0:k],eul_log['r'][0:k,2],label='Euler')
    ax1.plot(discrete_log['t'][0:k],discrete_log['r'][0:k,0],label='Discrete')
    ax2.plot(discrete_log['t'][0:k],discrete_log['r'][0:k,1],label='Discrete')
    ax3.plot(discrete_log['t'][0:k],discrete_log['r'][0:k,2],label='Discrete')
    ax1.plot(soln0['t'][0:k],soln0['r'][0:k,0],label='Soln')
    ax2.plot(soln0['t'][0:k],soln0['r'][0:k,1],label='Soln')
    ax3.plot(soln0['t'][0:k],soln0['r'][0:k,2],label='Soln')
    ax1.plot(ivp_result['t'][:],ivp_result['y'][0,:],label='IVP')
    ax2.plot(ivp_result['t'][:],ivp_result['y'][1,:],label='IVP')
    ax3.plot(ivp_result['t'][:],ivp_result['y'][2,:],label='IVP')
    fig1.suptitle(r'Position $\mathbf{r}^I$ (m)')
    ax1.set_ylabel(r'$r_x^I$')
    ax2.set_ylabel(r'$r_y^I$')
    ax3.set_ylabel(r'$r_z^I$')
    ax3.set_xlabel(r'Time')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(rk4_log['t'][0:k],rk4_log['v'][0:k,0],label='RK4')
    ax2.plot(rk4_log['t'][0:k],rk4_log['v'][0:k,1],label='RK4')
    ax3.plot(rk4_log['t'][0:k],rk4_log['v'][0:k,2],label='RK4')
    ax1.plot(eul_log['t'][0:k],eul_log['v'][0:k,0],label='Euler')
    ax2.plot(eul_log['t'][0:k],eul_log['v'][0:k,1],label='Euler')
    ax3.plot(eul_log['t'][0:k],eul_log['v'][0:k,2],label='Euler')
    ax1.plot(discrete_log['t'][0:k],discrete_log['v'][0:k,0],label='Discrete')
    ax2.plot(discrete_log['t'][0:k],discrete_log['v'][0:k,1],label='Discrete')
    ax3.plot(discrete_log['t'][0:k],discrete_log['r'][0:k,2],label='Discrete')
    ax1.plot(soln0['t'][0:k],soln0['v'][0:k,0],label='Soln')
    ax2.plot(soln0['t'][0:k],soln0['v'][0:k,1],label='Soln')
    ax3.plot(soln0['t'][0:k],soln0['v'][0:k,2],label='Soln')
    ax1.plot(ivp_result['t'][:],ivp_result['y'][3,:],label='IVP')
    ax2.plot(ivp_result['t'][:],ivp_result['y'][4,:],label='IVP')
    ax3.plot(ivp_result['t'][:],ivp_result['y'][5,:],label='IVP')
    fig2.suptitle(r'Velocity $\mathbf{v}^I$ (m/s)')
    ax1.set_ylabel(r'$v_x^I$')
    ax2.set_ylabel(r'$v_y^I$')
    ax3.set_ylabel(r'$v_z^I$')
    ax3.set_xlabel(r'Time')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig3, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(rk4_log['t'][0:k],rk4_log['T'][0:k,0],label='RK4')
    ax2.plot(rk4_log['t'][0:k],rk4_log['T'][0:k,1],label='RK4')
    ax3.plot(rk4_log['t'][0:k],rk4_log['T'][0:k,2],label='RK4')
    ax1.plot(eul_log['t'][0:k],eul_log['T'][0:k,0],label='Euler')
    ax2.plot(eul_log['t'][0:k],eul_log['T'][0:k,1],label='Euler')
    ax3.plot(eul_log['t'][0:k],eul_log['T'][0:k,2],label='Euler')
    ax1.plot(discrete_log['t'][0:k],discrete_log['T'][0:k,0],label='Discrete')
    ax2.plot(discrete_log['t'][0:k],discrete_log['T'][0:k,1],label='Discrete')
    ax3.plot(discrete_log['t'][0:k],discrete_log['T'][0:k,2],label='Discrete')
    ax1.plot(soln0['t'][0:k],soln0['T'][0:k,0],label='Soln')
    ax2.plot(soln0['t'][0:k],soln0['T'][0:k,1],label='Soln')
    ax3.plot(soln0['t'][0:k],soln0['T'][0:k,2],label='Soln')
    fig3.suptitle(r'Thrust')
    ax1.set_ylabel(r'$T_x$')
    ax2.set_ylabel(r'$T_y')
    ax3.set_ylabel(r'$T_z$')
    ax3.set_xlabel(r'Time')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

    print("fin")