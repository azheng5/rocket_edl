import math as m
import copy

import numpy             as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import traj_control
from tools import debug
from tools import attitude
from tools import plotting

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

        # Monte Carlo config
        self.num_runs = self.config['num_runs']

        # Time
        self.dt_sim = self.config['dt_sim']
        self.sim_hz = 1/self.dt_sim
        self.dt_c = self.config['dt_c']
        self.ctrl_hz = 1/self.dt_c
        self.max_tof = self.config["max_tof"]
        self.max_N = int(self.max_tof/self.dt_c)


        # Initialize runtime variables
        self.k = None # control iteration
        self.x = np.array([]) # state
        self.t = None # time
        self.tof_guess = None # current optimal tof guess

        # Log
        self.empty_log = {
            't': np.zeros(self.max_N+1),
            'r': np.zeros((self.max_N+1,3)), # position in inertial frame
            'v': np.zeros((self.max_N+1,3)), # velocity in inertial frame
            # 'qBI': np.zeros((self.max_N+1,4)), # quaternion from inertial to body
            # 'w_BI': np.zeros((self.max_N+1,3)), # angular velocity of body wrt inertial expressed in body
            'm': np.zeros(self.max_N+1),
            'T': np.zeros((self.max_N+1,3))
        }


    def run(self, mode):
        if mode == 'MIL':
            print('Running model-in-the-loop...')
            self.run_mil_monte_carlo()
        elif mode == 'SWIL':
            assert False, "SWIL not implemented yet, its coming soon!"
        elif mode == 'PIL':
            assert False, "PIL not implemented yet, its coming soon!"
        elif mode == 'HWIL':
            assert False, "I do not have enough money for that."
        else:
            assert False, "Invalid simulation mode"

    def run_mil_monte_carlo(self):
        '''Model-in-the-loop simulation driver'''

        # Allocate memory
        # Shallow copy doesn't work bc the numpy arrays within log does not create indep copies of these arrays
        # Thus when modifying an array in one list element, the array gets modified in the other list element as well
        # So need to use deep copy to ensure arrays in the dict are copied independently
        logs = [copy.deepcopy(self.empty_log) for i in range(self.config['num_runs'])]

        # Pregenerate monte carlo rocket config variables
        mc_configs = self.generate_monte_carlo_configs(self.config)

        # Run monte carlos
        for run_index in range(self.config["num_runs"]):
            log = self.run_mil(mc_configs[run_index])
            logs[run_index] = copy.deepcopy(log)

        # Plot trajectories
        self.plot(logs)

        return 1

    def run_mil(self, config):

        # Allocate memory for logged data (better than appending bc numpy arrays are immutable)
        log = self.empty_log.copy()

        # (Re-)initialize runtime variables
        # Need to re-initialize it every time this fn is called bc of monte carlo sim
        self.k = 0
        self.x = [config['r0'][0], config['r0'][1], config['r0'][2],
                  config['v0'][0], config['v0'][1], config['v0'][2],
                #   config['q0'][0], config['q0'][1], config['q0'][2], config['q0'][3],
                #   config['w0'][0], config['w0'][1], config['w0'][2],
                  config['m0']] # state
        self.t = 0 # time
        self.tof_guess = self.config["init_tof_guess"]
        terminate_sim = False

        # Enter simulation loop
        while not terminate_sim:

            # Trajectory control
            (soln, tof_guess_next) = traj_control.tof_search(self.x[0:3],self.x[3:6],self.x[6],self.tof_guess,config)

            T_des = soln['T'][0,:]

            # Clamp thrust norm
            if LA.norm(T_des) > self.config['T_max']:
                # warnings.warn('Optimal thrust violated max limit and was clamped')
                debug.warn('Optimal thrust violated max limit and was clamped')
                T_des = T_des * (self.config['T_max']/LA.norm(T_des))
            elif LA.norm(T_des) < self.config['T_min']:
                # warnings.warn('Optimal thrust violated min limit and was clamped')
                debug.warn('Optimal thrust violated max limit and was clamped')
                T_des = T_des * (self.config['T_min']/LA.norm(T_des))

            
            if self.k == 0:
                soln0 = soln
            print('Debug compare thrust',soln0['T'][self.k,:],T_des)
            

            # Attitude control
            # M_des = attitude_control.pid()

            # Control input
            u = T_des

            # Push current states to log
            log['t'][self.k] = self.t
            log['r'][self.k,:] = self.x[0:3]
            log['v'][self.k,:] = self.x[3:6]
            log['m'][self.k] = self.x[6]
            log['T'][self.k,:] = T_des

            # Propagate dynamics for current control step
            # TODO this will have to be restructured when incorporating attitude dynamics (maybe)
            for i in range( int(self.config["dt_c"]/self.config["dt_sim"]) ):
                x_next = self.rk4_step(self.eom,self.t,self.x,
                                       T_des)
                # x_next = self.discrete_eom(self.t,self.x,
                #                            T_des)
                # x_next = self.euler_step(self.eom, self.t,self.x,
                #                          T_des)
                self.x = x_next
                
                # Prepare subsequent dynamics step
                self.t += self.dt_sim

            # Prepare subsequent control step
            self.tof_guess = tof_guess_next
            self.k += 1
            terminate_sim = self.termination_conditions(T_des)

            print('k =',self.k)
            print('Debug compare state',soln0['r'][self.k,:],soln0['v'][self.k,:],soln0['m'][self.k],x_next)

        print("----------Results----------")
        print("Final position:",self.x[0:3])
        print("Final velocity:",self.x[3:6])
        print("Final mass:",self.x[6])

        # self.plot(log)

        return log
    
    # def eom(self, t, x, Tb, Mb, t_prof=0):
    #     """
    #     6-DoF equations of motion for rocket dynamics. Can be called using custom integrators 
    #     (i.e. rk4_step or euler_step) or built-in integrators (i.e. solve_ivp)

    #     Args:
    #         t: Current time.
    #         x (14,): Current state [rx ry rz vx vy vz qw qx qy qz wx wy wz m].
    #         Tb: Thrust in body frame. Can be either a single thrust command or a thrust profile. If 
    #         its a thrust profile, the current thrust will be determined from t_profile.
    #         Mb: Moment in body frame about the CG.
    #         t_prof (optional): Time profile associated with thrust profile.
        
    #     Returns:
    #         x_next: Predicted state one time step later.
            
    #     """

    #     # Extract state variables
    #     rx = x[0]
    #     ry = x[1]
    #     rz = x[2]
    #     vx = x[3]
    #     vy = x[4]
    #     vz = x[5]
    #     m = x[6]

    #     # Gravity
    #     # FG_b = np.array([-self.config["g"],0,0]).reshape((3,1)) # 3x1
    #     FG_b = m*self.config['g_i'].reshape((3,1))
    #     #TODO add variable gravity

    #     # Thrust
    #     if T.ndim == 2 and T.shape[0] > 1: # Force profile detection

    #         # NOTE: Don't interpolate, controls should not exceed the defined controller rate
    #         # t_index = np.abs(t_prof - t).argmin() # finds index of nearest time - nvm not necessary

    #         # Find index of largest time in the time profile that does not exceed t
    #         t_index = np.searchsorted(t_prof, t) - 1
            
    #         FT_b = T[t_index,:].reshape((3,1))

    #     elif T.shape == (3,): # Single force detection
    #         FT_b = T.reshape((3,1))

    #     # Compute state derivatives
    #     F_b = FG_b + FT_b
    #     v_dot = (1/m) * F_b
    #     m_dot = -self.config["alpha"]*LA.norm(FT_b)

    #     x_dot = np.array([vx, vy, vz, v_dot[0,0], v_dot[1,0], v_dot[2,0], m_dot])

    # return x_dot


    def eom(self, t, x, T, t_prof=0):
        """
        3-DoF equations of motion for rocket dynamics. Can be called using custom integrators 
        (i.e. rk4_step or euler_step) or built-in integrators (i.e. solve_ivp)

        Args:
            t: Current time.
            x (7,): Current state [rx ry rz vx vy vz m].
            T : Can be either a single thrust command or a thrust profile. If its a thrust profile,
            the current thrust will be determined from t_profile.
            t_prof (optional): Time profile associated with thrust profile.
        
        Returns:
            x_next: Predicted state one time step later.
            
        """

        # Extract state variables
        rx = x[0]
        ry = x[1]
        rz = x[2]
        vx = x[3]
        vy = x[4]
        vz = x[5]
        m = x[6]

        # Gravity
        # FG_b = np.array([-self.config["g"],0,0]).reshape((3,1)) # 3x1
        FG_b = m*self.config['g_i'].reshape((3,1))
        #TODO add variable gravity

        # Thrust
        if T.ndim == 2 and T.shape[0] > 1: # Force profile detection

            # NOTE: Don't interpolate, controls should not exceed the defined controller rate
            # t_index = np.abs(t_prof - t).argmin() # finds index of nearest time - nvm not necessary

            # Find index of largest time in the time profile that does not exceed t
            t_index = np.searchsorted(t_prof, t) - 1
            
            FT_b = T[t_index,:].reshape((3,1))

        elif T.shape == (3,): # Single force detection
            FT_b = T.reshape((3,1))

        # Compute state derivatives
        F_b = FG_b + FT_b
        v_dot = (1/m) * F_b
        m_dot = -self.config["alpha"]*LA.norm(FT_b)

        x_dot = np.array([vx, vy, vz, v_dot[0,0], v_dot[1,0], v_dot[2,0], m_dot])

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
        # print('sim term:',int(self.tof_guess/self.dt_c))
        if self.tof_guess <= self.dt_c:
            print("Simulation terminated: Expected TOF smaller than controller time step.")
            return True
        if self.x[0] <= 0:
            print("Simulation terminated: Glideslope constraint violated")
            return True
        if self.k > self.max_N:
            print("Simulation terminated: Max TOF reached")
            return True
        if self.x[6] <= self.config['m_dry']:
            print("Simulation terminated: Fuel usage depleted")
            return True
        # if LA.norm(T) > self.config['T_max']:
        #     print("Simulation terminated: Max thrust constraint violated")
        #     return True
        # if LA.norm(T) < self.config['T_min']:
        #     print("Simulation terminated: Min thrust constraint violated")
        #     return True

        return False
    
    def generate_monte_carlo_configs(self,config):
        
        # Allocate memory
        mc_config_template = config
        mc_configs = [mc_config_template.copy() for i in range(self.config['num_runs'])]

        for i in range(config['num_runs']):
            # Sample monte carlo variables
            r0 = np.random.normal(config['r0_mean'],config['r0_std'])
            v0 = np.random.normal(config['v0_mean'],config['v0_std'])

            # mc_configs[i] = config
            mc_configs[i]['r0'] = r0
            mc_configs[i]['v0'] = v0

        return mc_configs

    def plot(self,logs):
        #TODO create a general plotter that takes in some general data struct,
        # parses it, and plots standard things like posn, velo etc
        # current apps: convex mpc offline traj, real time controlled traj

        fig1 = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(self.num_runs):
            log = logs[i]
            ax.plot3D(log['r'][0:self.k,2],-log['r'][0:self.k,1],log['r'][0:self.k,0])
            ax.scatter(log['r'][0,2],-log['r'][0,1],log['r'][0,0],s=20)
            ax.scatter(log['r'][self.k-1,2],-log['r'][self.k-1,1],log['r'][self.k-1,0], marker='x')
        ax.scatter(0,0,0,marker='o',color='black',s=30)
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        ax.set_zlabel('x')
        ax.set_title('3d Trajectory')
        ax.set_box_aspect([1, 1, 1])

        #%%
        fig2, (ax1,ax2,ax3) = plt.subplots(3,1)
        for i in range(self.num_runs):
            log = logs[i]
            ax1.plot(log['t'][0:self.k],log['r'][0:self.k,0])
            ax2.plot(log['t'][0:self.k],log['r'][0:self.k,1])
            ax3.plot(log['t'][0:self.k],log['r'][0:self.k,2])
        fig2.suptitle(r'Position $\mathbf{r}^I$ (m)')
        ax1.set_ylabel(r'$r_x^I$')
        ax2.set_ylabel(r'$r_y^I$')
        ax3.set_ylabel(r'$r_z^I$')
        ax3.set_xlabel(r'Time')

        #%%
        fig3, (ax1,ax2,ax3) = plt.subplots(3,1)
        for i in range(self.num_runs):
            log = logs[i]
            ax1.plot(log['t'][0:self.k],log['v'][0:self.k,0])
            ax2.plot(log['t'][0:self.k],log['v'][0:self.k,1])
            ax3.plot(log['t'][0:self.k],log['v'][0:self.k,2])
        fig3.suptitle('Velocity')
        fig3.suptitle(r'Velocity $\mathbf{v}^I$ (m)')
        ax1.set_ylabel(r'$v_x^I$')
        ax2.set_ylabel(r'$v_y^I$')
        ax3.set_ylabel(r'$v_z^I$')
        ax3.set_xlabel(r'Time')

        #%%
        fig4, (ax1,ax2,ax3) = plt.subplots(3,1)
        for i in range(self.num_runs):
            log = logs[i]
            ax1.plot(log['t'][0:self.k],log['T'][0:self.k,0])
            ax2.plot(log['t'][0:self.k],log['T'][0:self.k,1])
            ax3.plot(log['t'][0:self.k],log['T'][0:self.k,2])
        fig4.suptitle('Thrust')
        ax1.set_ylabel(r'$T_x$')
        ax2.set_ylabel(r'$T_y')
        ax3.set_ylabel(r'$T_z$')
        ax3.set_xlabel(r'Time')

        #%%
        fig5, ax = plt.subplots()
        for i in range(self.num_runs):
            log = logs[i]
            ax.plot(log['t'][0:self.k],log['m'][0:self.k])
        fig5.suptitle('Mass')
        ax.set_ylabel('m (kg)')
        ax.set_xlabel('time (sec)')

        #%%
        fig6, ax = plt.subplots()
        for i in range(self.num_runs):
            log = logs[i]
            ax.plot(log['t'][0:self.k],LA.norm(log['T'][0:self.k,:],axis=1))
        fig6.suptitle('Thrust Norm')

        #%%
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print('fin.')

        print('fin2.')