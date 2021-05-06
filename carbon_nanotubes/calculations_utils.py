#Importing libraries
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

def gamma(omega_0, omega, mu, tau, T, R, d, m_max):
    """TPSE spectral density"""

    Pxx = P('x', omega, mu, tau, T, R, d, m_max)[0]*P('x', omega_0 - omega, mu, tau, T, R, d, m_max)[0]
    Pyy = P('y', omega, mu, tau, T, R, d, m_max)[0]*P('y', omega_0 - omega, mu, tau, T, R, d, m_max)[0]
    Pzz = P('z', omega, mu, tau, T, R, d, m_max)[0]*P('z', omega_0 - omega, mu, tau, T, R, d, m_max)[0]

    return (Pxx + Pyy + Pzz)/3
