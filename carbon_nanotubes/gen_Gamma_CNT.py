from calculations_utils import *

T = 300
R = 2*10**(-9)
mu_ev_array = np.array([0.25,0.5,0.75,1])
mob = 10**4
MCut = 20
m_max = 10
omegaCut = 0.0000044*e/hbar

try:
    Gammas = list(np.load('Gamma_CNT.npy'))
except:
    Gammas = []

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
for mu_ev in mu_ev_array[len(Gammas):]:
    print('mu (ev):', mu_ev)
    Gamma_mu = []
    for distance_nm in distances_nm:
        print(f'd (nm): {distance_nm}', end='\r')
        Gamma_mu.append(Gamma(d_matrix, ne, ng, mu_ev*e, tau_g(mob,mu_ev*e), T, R, R + distance_nm*10**(-9), m_max, MCut, omegaCut))
    Gammas.append(np.array(Gamma_mu))
    np.save('Gamma_CNT.npy',np.array(Gammas))

print('\nTodos os dados foram gerados com sucesso.')