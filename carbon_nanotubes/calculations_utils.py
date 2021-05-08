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

#-------------------------------------------CARBON NANOTUBES--------------------------------------------------------------------------------------
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

def r(m, sigma, omega, mu, tau, T, R, k):
    """Fresnel coefficient"""

    deltam = 1j*sigma*(m**2 + (k*R)**2)/(epsilon_0*omega*R)
    besselI = special.iv(m,k*R)
    result = -besselI**2*deltam/(1 + besselI*special.kn(m,k*R)*deltam)

    return result

def tau_g(mob, mu):
    
    return mob*mu/(e*10**16)

def dispersion_relation_approximated(m, kp, mu, R):
    num = e**2*mu*special.kn(m, kp*R)*special.iv(m, kp*R)*(m**2 + kp**2*R**2)
    den = np.pi*epsilon_0*R*hbar**2
    omega_p2 = num/den

    return np.sqrt(omega_p2)

#-------------------------------------------PURCELL FACTORS--------------------------------------------------------------------------------------
def int_z(sigma, omega, mu, tau, T, R, k, d, m_max):
    """Auxiliary function for Pz"""

    term0 = k**2*special.k0(k*d)**2*np.imag(r(0, sigma, omega, mu, tau, T, R, k))
    terms_array = np.array([k**2*special.kn(m,k*d)**2*np.imag(r(m, sigma, omega, mu, tau, T, R, k)) for m in range(1,m_max + 1)])
    result = term0 + 2*np.sum(terms_array)

    return result

def int_x(sigma, omega, mu, tau, T, R, k, d, m_max):
    """Auxiliary function for Px"""

    term0 = k**2*(special.kn(1,k*d) + special.kn(-1,k*d))**2*np.imag(r(0, sigma, omega, mu, tau, T, R, k))/4
    terms_array = np.array([k**2*(special.kn(m + 1,k*d) + special.kn(m - 1,k*d))**2*np.imag(r(m, sigma, omega, mu, tau, T, R, k))/4 for m in range(1,m_max + 1)])
    result = term0 + 2*np.sum(terms_array)

    return result

def int_y(sigma, omega, mu, tau, T, R, k, d, m_max):
    """Auxiliary function for Py"""

    terms_array = np.array([m**2*special.kn(m,k*d)**2*np.imag(r(m, sigma, omega, mu, tau, T, R, k))/d**2 for m in range(1,m_max + 1)])
    result = 2*np.sum(terms_array)

    return result

def P(dir, omega, mu, tau, T, R, d, m_max):
    """Purcell factor for a given direction"""

    sigma, _ = conductivity(omega,mu,tau,T)

    if dir == 'x':
        integrand = lambda k: int_x(sigma, omega, mu, tau, T, R, k, d, m_max)
    elif dir =='y':
        integrand = lambda k: int_y(sigma, omega, mu, tau, T, R, k, d, m_max)
    elif dir =='z':
        integrand = lambda k: int_z(sigma, omega, mu, tau, T, R, k, d, m_max)
    else:
        return None

    integral = integrate.quad(integrand, 0, 10/R)

    P = 1 - 3*c**3/(np.pi*omega**3)*integral[0]
    pct_error = 100*integral[1]/integral[0]
    
    return P, pct_error

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
def gamma(omega_0, omega, mu, tau, T, R, d, m_max):
    """TPSE spectral density"""

    Pxx = P('x', omega, mu, tau, T, R, d, m_max)[0]*P('x', omega_0 - omega, mu, tau, T, R, d, m_max)[0]
    Pyy = P('y', omega, mu, tau, T, R, d, m_max)[0]*P('y', omega_0 - omega, mu, tau, T, R, d, m_max)[0]
    Pzz = P('z', omega, mu, tau, T, R, d, m_max)[0]*P('z', omega_0 - omega, mu, tau, T, R, d, m_max)[0]

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

def Gamma(d_matrix, ne, ng, mu, tau, T, R, d, m_max, MCut, omegaCut):
    """ TPSE rate near a CNT"""
    integrand = lambda i: 2*gamma_0(d_matrix, ne,ng,i*domega(ne,ng),MCut)*gamma(domega(ne,ng), i*domega(ne,ng), mu, tau, T, R, d, m_max)/(10**10)
    integral = integrate.quad(integrand, omegaCut/domega(ne,ng),1/2, epsrel=10**(-4))

    return 10**10*domega(ne,ng)*integral[0]

