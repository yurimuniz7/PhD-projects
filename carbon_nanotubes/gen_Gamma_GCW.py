from calculations_utils import *

T = 300
mu = e
R_nm_array = np.array([20, 50, 100])
mob = 10**4
MCut = 20
m_max = 10
omegaCut = 0.0000044*e/hbar
epsilon = 11.68

try:
    Gammas = list(np.load('Gamma_GCW.npy'))
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
for R_nm in R_nm_array[len(Gammas):]:
    print('R (nm):', R_nm)
    Gamma_mu = []
    for distance_nm in distances_nm:
        print(f'd (nm): {distance_nm}', end='\r')
        Gamma_mu.append(Gamma(d_matrix, ne, ng, mu, tau_g(mob,mu), T, R_nm*10**-9, (R_nm + distance_nm)*10**(-9), m_max, MCut, omegaCut, epsilon=epsilon))
    Gammas.append(np.array(Gamma_mu))
    np.save('Gamma_GCW.npy',np.array(Gammas))

print('\nTodos os dados foram gerados com sucesso.')