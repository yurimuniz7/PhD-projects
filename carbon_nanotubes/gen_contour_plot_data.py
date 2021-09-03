from calculations_utils import *

omega_0 = domega(6,5)
mob = 10**4
mu = e
T = 300
epsilon = 11.68

# DON'T FORGET TO CHANGE THE RADIUS ARRAY
R_nm_array = np.arange(20,60,0.5)

distance_nm = 10
m_max = 12

gammas = []
omega_steps = np.arange(0.002,0.5001,0.002)
for R_nm in R_nm_array:
    print('R (nm):', R_nm)
    gamma_R = []
    for step in omega_steps:
        gamma_R.append(gamma(omega_0, omega_0*step, mu, tau_g(mob,mu), T, R_nm*10**(-9), (distance_nm + R_nm)*10**(-9), m_max, epsilon = epsilon))
    gammas.append(gamma_R)

    # DON'T FORGET TO GIVE A PROPER NAME TO THE DATA
    np.save('contour_plot_data_20_60.npy',np.array(gammas))