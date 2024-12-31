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

        tof_guess = self.init_tof_guess


        # Allocate memory for logged data (better than appending bc numpy arrays are immutable)
        log = {
            't': np.zeros(self.max_N),
            'r': np.zeros((self.max_N,3)),
            'v': np.zeros((self.max_N,3)),
            'm': np.zeros(self.max_N),
            'T': np.zeros((self.max_N,3))
        }

        while self.x[2] > 0:
            #TODO add termination conditions function (thrust limit exceeded, AOA exceeded, etc)

            # Trajectory control
            (soln, tof_guess) = traj_control.tof_search(self.x[0:3],self.x[3:6],self.x[6],tof_guess,self.config)
            T_des = soln['T'][0,:]

            # Attitude control
            # M_des = attitude_control.pid()

            # Control input
            u = T_des


            # Propagate dynamics for current control step
            # TODO this will have to be restructured when incorporating attitude dynamics (maybe)
            for i in range( int(self.config["dt_c"]/self.config["dt_sim"]) ):
                x_next = self.rk4_step(self.eom,self.t,self.x,u)
                self.x = x_next
                
                self.t += self.dt_sim

            # Push data to log
            if self.k == 0:
                log['t'][0] = 0
                log['r'][0,:] = self.config['r0']
                log['v'][0,:] = self.config['v0']
                log['m'][0] = self.config['m0']
                log['T'][0,:] = T_des
            else:
                log['t'][self.k] = self.t
                log['r'][self.k,:] = self.x[0:3]
                log['v'][self.k,:] = self.x[3:6]
                log['m'][self.k] = self.x[6]
                log['T'][self.k,:] = T_des

            self.k += 1

        self.plot(log)


        return 1


    def eom(self, t, x, u):

        # Extract state variables
        rx = x[0]
        ry = x[1]
        rz = x[2]
        vx = x[3]
        vy = x[4]
        vz = x[5]
        m = x[6]
        T_des = u

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


    def rk4_step(self, fn, t, x, u):

        k1 = fn(t,x,u)
        k2 = fn(t + self.dt_sim/2, x + (self.dt_sim/2)*k1, u)
        k3 = fn(t + self.dt_sim/2, x + (self.dt_sim/2)*k2, u)
        k4 = fn(t + self.dt_sim/2, x + self.dt_sim*k3, u)

        return x + (self.dt_sim/6)*(k1 + 2*k2 + 2*k3 + k4)
    

    def plot(self,log):
        

        fig1 = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(log['r'][:,2],log['r'][:,1],log['r'][:,0])
        ax.scatter(log['r'][0,2],log['r'][0,1],log['r'][0,0],s=20)
        ax.scatter(log['r'][-1,2],log['r'][-1,1],log['r'][-1,0], marker='x',color='blue')
        ax.scatter(0,0,0,marker='x',color='red',s=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3d Trajectory')

        #%%
        fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(log['t'],log['r'][:,0])
        ax2.plot(log['t'],log['r'][:,1])
        ax3.plot(log['t'],log['r'][:,2])
        fig2.suptitle('Position')

        #%%
        fig3, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(log['t'],log['v'][:,0])
        ax2.plot(log['t'],log['v'][:,1])
        ax3.plot(log['t'],log['v'][:,2])
        fig3.suptitle('Velocity')

        #%%
        fig4, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(log['t'][0:-1],log['T'][:,0])
        ax2.plot(log['t'][0:-1],log['T'][:,1])
        ax3.plot(log['t'][0:-1],log['T'][:,2])
        fig4.suptitle('Thrust')

        #%%
        fig5, ax = plt.subplots()
        ax.plot(log['t'],log['m'])
        fig5.suptitle('Mass')

        #%%
        fig6, ax = plt.subplots()
        ax.plot(log['t'][0:-1],LA.norm(log['T'],axis=1))
        fig6.suptitle('Thrust Norm')

        #%%
        plt.show()

        print('fin.')