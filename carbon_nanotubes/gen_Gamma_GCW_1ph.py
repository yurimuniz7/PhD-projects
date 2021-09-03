from calculations_utils import *

T = 300
mu = e
R_nm_array = np.array([20, 50, 100])
mob = 10**4
epsilon = 11.68
MCut = 10
m_max = 10
omegaCut = 0.0000044*e/hbar

ne = 6
ng = 5
d_matrix = []
dne = []
dng = []
for m in range(2,MCut + 1):
    dne.append(d(ne,m))
    dng.append(d(ng,m))

d_matrix.append(dne)
d_matrix.append(dng)

distances_nm = np.logspace(0,3,61)
Gamma065 = Gamma01ph(6,5)
Gamma064 = Gamma01ph(6,4)
Gamma063 = Gamma01ph(6,3)
Gamma062 = Gamma01ph(6,2)
Gammas = []
for R_nm in R_nm_array:
    print('R (nm):', R_nm)
    Gamma_R = []
    for distance_nm in distances_nm:
        print('d (nm):', distance_nm, end='\r')
        Gamma65 = Gamma065*P_iso(domega(6,5), mu, tau_g(mob,mu), T, R_nm*10**-9, (R_nm + distance_nm)*10**(-9), m_max, epsilon)
        Gamma64 = Gamma064*P_iso(domega(6,4), mu, tau_g(mob,mu), T, R_nm*10**-9, (R_nm + distance_nm)*10**(-9), m_max, epsilon)
        Gamma63 = Gamma063*P_iso(domega(6,3), mu, tau_g(mob,mu), T, R_nm*10**-9, (R_nm + distance_nm)*10**(-9), m_max, epsilon)
        Gamma62 = Gamma062*P_iso(domega(6,2), mu, tau_g(mob,mu), T, R_nm*10**-9, (R_nm + distance_nm)*10**(-9), m_max, epsilon)
        Gamma_R.append(Gamma65 + Gamma64 + Gamma63 + Gamma62)
    Gammas.append(Gamma_R)

np.save('Gamma_GCW_1ph.npy',np.array(Gammas))