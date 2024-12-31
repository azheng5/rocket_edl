import numpy as np

dt_c = 1
m_fuel = 10000
m_dry = 25600
m0 = m_fuel + m_dry
g = 9.807
g_i = np.array([0,0,-g])
ex =  np.array([1,0,0])
ey =  np.array([0,1,0])
ez =  np.array([0,0,1])
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
dt_sim = 0.1
init_tof_guess = 50
max_tof = 60 # if tof is higher than this somethings wrong

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
    "r0": r0,
    "v0": v0
}