#Importing libraries
import math
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import special

# Defining some important constants
hbar = (6.62607*10**(-34))/(2*np.pi)
e = 1.602177*10**(-19)
epsilon_0 = 8.85419*10**(-12)
mu_0 = 4*np.pi*10**(-7)
c = np.sqrt(1/(epsilon_0*mu_0))
kb = 1.38*10**(-23)
me = 9.10938356*10**(-31)
a0 = 4*np.pi*epsilon_0*hbar**2/(me*e**2)

#-------------------------------------------CARBON NANOTUBES and GRAPHENE COATED WIRES--------------------------------------------------------------------------------------
def G(x,mu,T):
    """Auxiliary function for graphene conductivity"""
    den1 = np.cosh(mu/(kb*T))/np.sinh(x/(kb*T))
    den2 = np.tanh(x/(kb*T))

    return 1/(den1 + den2)

def conductivity(omega, mu, tau, T):
    """Graphene conductivity"""

    term1_prefactor = (2j*e**2*kb*T/((omega + 1j/tau)*np.pi*hbar**2))
    term1 = term1_prefactor*np.log(2*np.cosh(mu/(2*kb*T)))
    integrand = lambda x: (G(hbar*omega*x/2,mu,T) - G(hbar*omega/2,mu,T))/(1 - x**2)
    integral = integrate.quad(integrand, 0, 10)

    sigma = term1 + (e**2/(4*hbar))*(G(hbar*omega/2,mu,T) + 2j*integral[0]/np.pi)
    pct_error = 100*integral[1]/integral[0]

    return sigma, pct_error

def tau_g(mob, mu):
    
    return mob*mu/(e*10**16)


def r(m, sigma, omega, mu, tau, T, R, k, epsilon = 1):
    """Fresnel coefficient"""

    deltam = 1j*sigma*(m**2 + (k*R)**2)/(epsilon_0*omega*R)
    besselI = special.iv(m,k*R)
    term_epsilon_num = k*R*(epsilon - 1)*besselI*special.ivp(m,k*R)
    term_epsilon_den = (epsilon*special.kn(m,k*R)*special.ivp(m,k*R) - besselI*special.kvp(m,k*R))*k*R
    result = -(term_epsilon_num + besselI**2*deltam)/(term_epsilon_den + besselI*special.kn(m,k*R)*deltam)

    return result

def r_cnt(m, sigma, omega, mu, tau, T, R, k):
    """Fresnel coefficient for CNTs only"""

    deltam = 1j*sigma*(m**2 + (k*R)**2)/(epsilon_0*omega*R)
    besselI = special.iv(m,k*R)
    result = -besselI**2*deltam/(1 + besselI*special.kn(m,k*R)*deltam)

    return result

def F(n, m, sigma, omega, mu, tau, T, R, range, theta, epsilon = 1):
    """Auxiliary function for calculating the dispersion relation through integration in the complex plane"""

    phase = np.exp(1j*theta)
    k = (1 + 0.99*phase)*range*omega/c
    dk = 0.99j*range*phase*omega/c
    deltam = 1j*sigma*(m**2 + (k*R)**2)/(epsilon_0*omega*R)
    besselI = special.iv(m,k*R)
    term_epsilon_den = (epsilon*special.kv(m,k*R)*special.ivp(m,k*R) - besselI*special.kvp(m,k*R))*k*R
    return dk*k**n/(term_epsilon_den + besselI*special.kv(m,k*R)*deltam)


def IF(n, m, sigma, omega, mu, tau, T, R, range, epsilon = 1):
    """Integral in the complex plane"""

    F_re = lambda theta: np.real(F(n, m, sigma, omega, mu, tau, T, R, range, theta, epsilon))
    F_im = lambda theta: np.imag(F(n, m, sigma, omega, mu, tau, T, R, range, theta, epsilon))

    integral_re = integrate.quad(F_re, 0, 2*np.pi)
    integral_im = integrate.quad(F_im, 0, 2*np.pi)
    result = integral_re[0] + 1j*integral_im[0]

    return result

def kp_res(m, sigma, omega, mu, tau, T, R, range, epsilon = 1):
    """Plasmon wavevector calculation. Returns zero if there is no solution for the transcedental equation"""

    I0 = IF(0, m, sigma, omega, mu, tau, T, R, range, epsilon)
    I1 = IF(1, m, sigma, omega, mu, tau, T, R, range, epsilon)
    I2 = IF(2, m, sigma, omega, mu, tau, T, R, range, epsilon)

    if (I0 == 0)|(I1 == 0)|(I2 == 0):
        return 0

    result1 = I1/I0
    result2 = I2/I1

    pct_error = 100*np.abs((result2 - result1)/result1)
    if pct_error > 1:
        return 0
    else:
        return result1

def dispersion_relation_approximated(m, kp, mu, R, epsilon = 1):
    """Approximated expression for the dispersion relation (not inverted)"""

    num = e**2*mu*special.kn(m, kp*R)*special.iv(m, kp*R)*(m**2 + kp**2*R**2)
    besselI = special.iv(m,kp*R)
    term_epsilon_den = (epsilon*special.kn(m,kp*R)*special.ivp(m,kp*R) - besselI*special.kvp(m,kp*R))*kp*R
    den = term_epsilon_den*np.pi*epsilon_0*R*hbar**2
    omega_p2 = num/den

    return np.sqrt(omega_p2)

def omega_min(m,mu,R,epsilon):
    """Minimum frequency of excitation of a plasmonic mode"""
    num = e**2*mu*m
    den = (1 + epsilon)*np.pi*epsilon_0*hbar**2*R
    return np.sqrt(num/den)

#-------------------------------------------PURCELL FACTORS--------------------------------------------------------------------------------------
def int_z(sigma, omega, mu, tau, T, R, k, d, m_max, epsilon=1):
    """Auxiliary function for Pz"""

    term0 = k**2*special.k0(k*d)**2*np.imag(r(0, sigma, omega, mu, tau, T, R, k, epsilon))
    terms_array = np.array([k**2*special.kn(m,k*d)**2*np.imag(r(m, sigma, omega, mu, tau, T, R, k, epsilon)) for m in range(1,m_max + 1)])
    result = term0 + 2*np.sum(terms_array)

    return result

def int_x(sigma, omega, mu, tau, T, R, k, d, m_max, epsilon=1):
    """Auxiliary function for Px"""
    term0 = k**2*(special.kn(1,k*d) + special.kn(-1,k*d))**2*np.imag(r(0, sigma, omega, mu, tau, T, R, k, epsilon))/4
    terms_array = np.array([k**2*(special.kn(m + 1,k*d) + special.kn(m - 1,k*d))**2*np.imag(r(m, sigma, omega, mu, tau, T, R, k, epsilon))/4 for m in range(1,m_max + 1)])
    result = term0 + 2*np.sum(terms_array)

    return result

def int_y(sigma, omega, mu, tau, T, R, k, d, m_max, epsilon=1):
    """Auxiliary function for Py"""

    terms_array = np.array([m**2*special.kn(m,k*d)**2*np.imag(r(m, sigma, omega, mu, tau, T, R, k, epsilon))/d**2 for m in range(1,m_max + 1)])
    result = 2*np.sum(terms_array)

    return result

def P(dir, sigma, omega, mu, tau, T, R, d, m_max, epsilon=1):
    """Purcell factor for a given direction"""

    if dir == 'x':
        integrand = lambda k: int_x(sigma, omega, mu, tau, T, R, k, d, m_max, epsilon)
    elif dir =='y':
        integrand = lambda k: int_y(sigma, omega, mu, tau, T, R, k, d, m_max, epsilon)
    elif dir =='z':
        integrand = lambda k: int_z(sigma, omega, mu, tau, T, R, k, d, m_max, epsilon)
    else:
        return None

    integral = integrate.quad(integrand, 0, 10/R)

    P = 1 - 3*c**3/(np.pi*omega**3)*integral[0]
    pct_error = 100*integral[1]/integral[0]
    
    return P, pct_error

def P_iso(omega, mu, tau, T, R, d, m_max, epsilon=1):
    """TPSE spectral density"""

    sigma, _ = conductivity(omega,mu,tau,T)
    Px = P('x', sigma, omega, mu, tau, T, R, d, m_max, epsilon)[0]
    Py = P('y', sigma, omega, mu, tau, T, R, d, m_max, epsilon)[0]
    Pz = P('z', sigma, omega, mu, tau, T, R, d, m_max, epsilon)[0]

    return (Px + Py + Pz)/3

#-------------------------------------------HYDROGEN FUNCTIONS--------------------------------------------------------------------------------------
def R(n, l, r):
    """Hydrogen radial function"""
    prefactor = np.sqrt(4*math.factorial(n - l - 1)/(n**4*math.factorial(n + l)))*(2*r/n)**l

    return prefactor*special.eval_genlaguerre(n - l - 1, 2*l + 1, 2*r/n)*np.exp(-r/n)

def domega(n,m):
    """Hydrogen frequencies"""
    prefactor = me*e**4/(32*np.pi**2*epsilon_0**2*hbar**3)

    return prefactor*(m**(-2) - n**(-2))

def d(n,m):
    """Hydrogen transition dipole moments"""
    integrand = lambda r: r**3*R(n,0,r)*R(m,1,r)
    integral = integrate.quad(integrand, 0, np.inf)

    return e*a0*integral[0]/np.sqrt(3)

def Gamma01ph(ne,ng):
    """One-photon SE rate in free space"""

    return d(ne,ng)**2*domega(ne,ng)**3/(3*np.pi*epsilon_0*hbar*c**3)

def T(d_matrix, ne, ng, omega, MCut):
    """TPSE transition tensor"""
    summands = []

    for m in list(range(2,ng+1)) + list(range(ne,MCut + 1)):
        in_brackets = (domega(ne,m) - omega)**-1 + (domega(ne,m) - (domega(ne,ng) - omega))**-1
        summands.append(d_matrix[0][m-2]*d_matrix[1][m-2]*in_brackets)

    return np.sum(summands)

#-------------------------------------------TPSE SPECTRAL DENSITIES--------------------------------------------------------------------------------------
def gamma(omega_0, omega, mu, tau, T, R, d, m_max, epsilon=1):
    """TPSE spectral density"""
    sigma1, _ = conductivity(omega,mu,tau,T)
    sigma2, _ = conductivity(omega_0 - omega,mu,tau,T)

    Pxx = P('x', sigma1, omega, mu, tau, T, R, d, m_max, epsilon)[0]*P('x', sigma2, omega_0 - omega, mu, tau, T, R, d, m_max, epsilon)[0]
    Pyy = P('y', sigma1, omega, mu, tau, T, R, d, m_max, epsilon)[0]*P('y', sigma2, omega_0 - omega, mu, tau, T, R, d, m_max, epsilon)[0]
    Pzz = P('z', sigma1, omega, mu, tau, T, R, d, m_max, epsilon)[0]*P('z', sigma2, omega_0 - omega, mu, tau, T, R, d, m_max, epsilon)[0]

    return (Pxx + Pyy + Pzz)/3

def gamma_0(d_matrix, ne, ng, omega, MCut):
    """Free space spectral density for hydrogen"""
    prefactor = mu_0**2/(12*np.pi**3*hbar**2*c**2)

    return prefactor*omega**3*(domega(ne,ng) - omega)**3*np.abs(T(d_matrix, ne,ng,omega,MCut))**2


#-------------------------------------------TPSE RATE AND QUANTUM EFFICIENCIES--------------------------------------------------------------------------------------
def Gamma_0(d_matrix, ne, ng, MCut, omegaCut):
    """Free space TPSE rate"""
    integrand = lambda w: 2*gamma_0(d_matrix, ne,ng,w,MCut)
    integral = integrate.quad(integrand, omegaCut, domega(ne,ng)/2)

    return integral[0]

def Gamma(d_matrix, ne, ng, mu, tau, T, R, d, m_max, MCut, omegaCut, epsilon=1):
    """ TPSE rate near a CNT"""
    integrand = lambda i: 2*gamma_0(d_matrix, ne,ng,i*domega(ne,ng),MCut)*gamma(domega(ne,ng), i*domega(ne,ng), mu, tau, T, R, d, m_max, epsilon)/(10**10)
    integral = integrate.quad(integrand, omegaCut/domega(ne,ng),1/2, epsrel=10**(-4))

    return 10**10*domega(ne,ng)*integral[0]

