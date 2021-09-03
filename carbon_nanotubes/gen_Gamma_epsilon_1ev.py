from calculations_utils import *

mob = 10**4
mu = e
T = 300
R = 50*10**(-9)
distance = 10*10**-9
m_max = 12
MCut = 20
m_max = 10
omegaCut = 0.0000044*e/hbar

try:
    Gammas = list(np.load('Gamma_GCW_epsilon_1ev.npy'))
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

epsilon_array = np.arange(2,10.0001,0.01)
for epsilon in epsilon_array[len(Gammas):]:
    print(f'epsilon: {epsilon}', end='\r')
    Gammas.append(Gamma(d_matrix, ne, ng, mu, tau_g(mob,mu), T, R, R + distance, m_max, MCut, omegaCut, epsilon=epsilon))
    np.save('Gamma_GCW_epsilon_1ev.npy',np.array(Gammas))

print('\nTodos os dados foram gerados com sucesso.')