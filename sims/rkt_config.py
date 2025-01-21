import numpy as np

dt_c = 1
m_fuel = 10000
m_dry = 25600
m0 = m_fuel + m_dry
g = 9.807
g_i = np.array([-g,0,0])
ex =  np.array([1,0,0])
ey =  np.array([0,1,0])
ez =  np.array([0,0,1])
Isp = 311
T_max = 411000
T_min = 0.4*T_max
alpha = 1/(Isp*g)
glideslope = 1*(np.pi/180)
thrust_cone = 10*(np.pi/180)
r0_mean = np.array([2000,0,0])
v0_mean = np.array([-50,0,0])
r0_std = np.array([50,100,100]) # 50 100 100
v0_std = np.array([5,5,5]) # 5 5 5
v_max = 1000000 # arb
v_horiz_max = 1000000 # arb
dt_sim = 0.01
init_tof_guess = 50
max_tof = 100 # if tof is higher than this somethings wrong
num_runs = 1

config = {
    "m_fuel": m_fuel,
    "m_dry": m_dry,
    "m0": m0,
    "g": g,
    "g_i": g_i,
    "ex": ex,
    "ey": ey,
    "ez": ez,
    "Isp": Isp,
    "T_max": T_max,
    "T_min": T_min,
    "alpha": alpha,
    "glideslope": glideslope,
    "thrust_cone": thrust_cone,
    "v_max": v_max,
    "v_horiz_max":v_horiz_max,
    "dt_c": dt_c,
    "dt_sim": dt_sim,
    "init_tof_guess": init_tof_guess,
    "max_tof": max_tof,
    "r0_mean": r0_mean,
    "v0_mean": v0_mean,
    "r0_std": r0_std,
    "v0_std": v0_std,
    "r0": np.array([0,0,0]),
    "v0": np.array([0,0,0]),
    "num_runs": num_runs
}