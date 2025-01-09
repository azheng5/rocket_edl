import math as m

import numpy             as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import traj_control

# def null_config():
#     """Initialize config for Rocket object"""

#     return {

#     }

class Rocket:

    def __init__(self, config):
        # self.config = null_config()
        # for key in config.keys():
        #     self.config[key] = config[key]
        self.config = config

        # Initial condition
        r0 = self.config['r0']
        v0 = self.config['v0']
        m0 = self.config['m0']
        self.x = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], m0] # state
        self.t = 0 # time

        # Time
        self.dt_sim = self.config['dt_sim']
        self.sim_hz = 1/self.dt_sim
        self.dt_c = self.config['dt_c']
        self.ctrl_hz = 1/self.dt_c
        self.max_tof = self.config["max_tof"]
        self.max_N = int(self.max_tof/self.dt_c)
        self.k = 0 # current iteration


        # Trajectory control config
        self.init_tof_guess = self.config["init_tof_guess"]

        # Attitude control config


    def run(self, mode):
        if mode == 'MIL':
            print('Running model-in-the-loop...')
            self.run_mil()
        elif mode == 'SWIL':
            assert False, "SWIL not implemented yet, its coming soon!"
        elif mode == 'PIL':
            assert False, "PIL not implemented yet, its coming soon!"
        elif mode == 'HWIL':
            assert False, "I do not have enough money for that."
        else:
            assert False, "Invalid simulation mode"

    def run_mil(self):
        '''Model-in-the-loop simulation driver'''

        # Allocate memory for logged data (better than appending bc numpy arrays are immutable)
        log = {
            't': np.zeros(self.max_N+1),
            'r': np.zeros((self.max_N+1,3)),
            'v': np.zeros((self.max_N+1,3)),
            'm': np.zeros(self.max_N+1),
            'T': np.zeros((self.max_N+1,3))
        }

        # Initialize simulation variables
        tof_guess = self.init_tof_guess
        terminate_sim = False

        # Enter simulation loop
        while not terminate_sim:

            # Trajectory control
            (soln, tof_guess) = traj_control.tof_search(self.x[0:3],self.x[3:6],self.x[6],tof_guess,self.config)
            T_des = soln['T'][0,:]
            
            if self.k == 0:
                soln0 = soln
            print('Debug compare thrust',soln0['T'][self.k,:],T_des)
            
            

            # Attitude control
            # M_des = attitude_control.pid()

            # Control input
            u = T_des

            # Push current states to log
            # if self.k == 0:
            #     log['t'][0] = 0
            #     log['r'][0,:] = self.config['r0']
            #     log['v'][0,:] = self.config['v0']
            #     log['m'][0] = self.config['m0']
            #     log['T'][0,:] = T_des
            log['t'][self.k] = self.t
            log['r'][self.k,:] = self.x[0:3]
            log['v'][self.k,:] = self.x[3:6]
            log['m'][self.k] = self.x[6]
            log['T'][self.k,:] = T_des

            # Propagate dynamics for current control step
            # TODO this will have to be restructured when incorporating attitude dynamics (maybe)
            for i in range( int(self.config["dt_c"]/self.config["dt_sim"]) ):
                # x_next = self.euler_step(self.eom,self.t,self.x,T_des)
                x_next = self.discrete_eom(self.t,self.x,T_des)
                self.x = x_next


                
                # Prepare subsequent dynamics step
                self.t += self.dt_sim

            # Prepare subsequent control step
            self.k += 1
            terminate_sim = self.termination_conditions(T_des)

            print('k =',self.k)
            print('Debug compare state',soln0['r'][self.k,:],soln0['v'][self.k,:],soln0['m'][self.k],x_next)

        self.plot(log)

        return 1


    def eom(self, t, x, T_des):

        # Extract state variables
        rx = x[0]
        ry = x[1]
        rz = x[2]
        vx = x[3]
        vy = x[4]
        vz = x[5]
        m = x[6]

        # Gravity
        FG_b = np.array([-self.config["g"],0,0])

        # Thrust
        FT_b = T_des

        # Calculate state derivatives
        F_b = FG_b + FT_b
        v_dot = (1/m) * F_b
        m_dot = -self.config["alpha"]*LA.norm(FT_b)

        x_dot = np.array([vx, vy, vz, v_dot[0], v_dot[1], v_dot[2], m_dot])

        return x_dot
    
    def discrete_eom(self,t,x,T_des):

        # Extract state variables
        rx = x[0]
        ry = x[1]
        rz = x[2]
        vx = x[3]
        vy = x[4]
        vz = x[5]
        m = x[6]

        rx_next = rx + vx*self.config['dt_sim'] + 0.5 * (self.config['dt_sim']**2) * (self.config['g_i'][0] + T_des[0]/m)
        ry_next = ry + vy*self.config['dt_sim'] + 0.5 * (self.config['dt_sim']**2) * (self.config['g_i'][1] + T_des[1]/m)
        rz_next = rz + vz*self.config['dt_sim'] + 0.5 * (self.config['dt_sim']**2) * (self.config['g_i'][2] + T_des[2]/m)
        vx_next = vx + self.config['dt_sim'] * (self.config['g_i'][0] + T_des[0]/m)
        vy_next = vy + self.config['dt_sim'] * (self.config['g_i'][1] + T_des[1]/m)
        vz_next = vz + self.config['dt_sim'] * (self.config['g_i'][2] + T_des[2]/m)
        m_next = m - self.config['dt_sim'] * self.config['alpha'] * LA.norm(T_des)

        x_next = np.array([rx_next, ry_next, rz_next, vx_next, vy_next, vz_next, m_next])
    
        return x_next
        # rx_dynamics = r[k+1,0] == r[k,0] + v[k,0]*dt_c + 0.5 * (dt_c**2) * (g_i[0] + u[k,0])
        # ry_dynamics = r[k+1,1] == r[k,1] + v[k,1]*dt_c + 0.5 * (dt_c**2) * (g_i[1] + u[k,1])
        # rz_dynamics = r[k+1,2] == r[k,2] + v[k,2]*dt_c + 0.5 * (dt_c**2) * (g_i[2] + u[k,2])
        # vx_dynamics = v[k+1,0] == v[k,0] + dt_c * (g_i[0] + u[k,0])
        # vy_dynamics = v[k+1,1] == v[k,1] + dt_c * (g_i[1] + u[k,1])
        # vz_dynamics = v[k+1,2] == v[k,2] + dt_c * (g_i[2] + u[k,2])

        # z_dynamics = z[k+1] == z[k] - alpha*sigma[k]*dt_c


    def rk4_step(self, fn, t, x, T_des):

        k1 = fn(t,x,T_des)
        k2 = fn(t + self.dt_sim/2, x + (self.dt_sim/2)*k1, T_des)
        k3 = fn(t + self.dt_sim/2, x + (self.dt_sim/2)*k2, T_des)
        k4 = fn(t + self.dt_sim, x + self.dt_sim*k3, T_des)

        return x + (self.dt_sim/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    def euler_step(self,fn,t,x,T_des):

        xdot = fn(t,x,T_des)

        return x + self.dt_sim*xdot
    
    def termination_conditions(self, T):
        if self.x[0] < 0:
            print("Simulation terminated: Glideslope constraint violated")
            return True
        if self.k > self.max_N:
            print("Simulation terminated: Max TOF reached")
            return True
        if self.x[6] <= self.config['m_dry']:
            print("Simulation terminated: Fuel usage depleted")
            return True
        if LA.norm(T) > self.config['T_max'] + 1000:
            print("Simulation terminated: Max thrust constraint violated")
            return True
        if LA.norm(T) < self.config['T_min'] - 1000:
            print("Simulation terminated: Min thrust constraint violated")
            return True

        return False
        
    

    def plot(self,log):
        #TODO create a general plotter that takes in some general data struct,
        # parses it, and plots standard things like posn, velo etc
        # current apps: convex mpc offline traj, real time controlled traj


        fig1 = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(log['r'][0:self.k,0],log['r'][0:self.k,1],log['r'][0:self.k,2])
        ax.scatter(log['r'][0,0],log['r'][0,1],log['r'][0,2],s=20)
        ax.scatter(log['r'][self.k-1,0],log['r'][self.k-1,1],log['r'][self.k-1,2], marker='x',color='blue')
        ax.scatter(0,0,0,marker='x',color='red',s=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3d Trajectory')

        #%%
        fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(log['t'][0:self.k],log['r'][0:self.k,0])
        ax2.plot(log['t'][0:self.k],log['r'][0:self.k,1])
        ax3.plot(log['t'][0:self.k],log['r'][0:self.k,2])
        fig2.suptitle('Position')

        #%%
        fig3, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(log['t'][0:self.k],log['v'][0:self.k,0])
        ax2.plot(log['t'][0:self.k],log['v'][0:self.k,1])
        ax3.plot(log['t'][0:self.k],log['v'][0:self.k,2])
        fig3.suptitle('Velocity')

        #%%
        fig4, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(log['t'][0:self.k],log['T'][0:self.k,0])
        ax2.plot(log['t'][0:self.k],log['T'][0:self.k,1])
        ax3.plot(log['t'][0:self.k],log['T'][0:self.k,2])
        fig4.suptitle('Thrust')

        #%%
        fig5, ax = plt.subplots()
        ax.plot(log['t'][0:self.k],log['m'][0:self.k])
        fig5.suptitle('Mass')

        #%%
        fig6, ax = plt.subplots()
        ax.plot(log['t'][0:self.k],LA.norm(log['T'][0:self.k,:],axis=1))
        fig6.suptitle('Thrust Norm')

        #%%
        plt.show()

        print('fin.')