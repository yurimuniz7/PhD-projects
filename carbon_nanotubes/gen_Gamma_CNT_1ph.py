from calculations_utils import *

T = 300
R = 2*10**(-9)
mu_ev_array = np.array([0.25,0.5,0.75,1])
mob = 10**4
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
for mu_ev in mu_ev_array:
    print('mu (ev):', mu_ev)
    Gamma_mu = []
    for distance_nm in distances_nm:
        print('d (nm):', distance_nm, end='\r')
        Gamma65 = Gamma065*P_iso(domega(6,5), mu_ev*e, tau_g(mob,mu_ev*e), T, R, R + distance_nm*10**(-9), m_max)
        Gamma64 = Gamma064*P_iso(domega(6,4), mu_ev*e, tau_g(mob,mu_ev*e), T, R, R + distance_nm*10**(-9), m_max)
        Gamma63 = Gamma063*P_iso(domega(6,3), mu_ev*e, tau_g(mob,mu_ev*e), T, R, R + distance_nm*10**(-9), m_max)
        Gamma62 = Gamma062*P_iso(domega(6,2), mu_ev*e, tau_g(mob,mu_ev*e), T, R, R + distance_nm*10**(-9), m_max)
        Gamma_mu.append(Gamma65 + Gamma64 + Gamma63 + Gamma62)
    Gammas.append(Gamma_mu)

np.save('Gamma_CNT_1ph.npy',np.array(Gammas))