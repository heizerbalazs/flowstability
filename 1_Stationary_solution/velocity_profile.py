import numpy as np
from scipy.integrate import odeint, trapz
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

class VelocityProfile:
    """
        This class represents a velocity profile.

        parameters:
        U - the velocity profile
        dU - first derivative of the velocity profile
        d2U - second derivative of the velocity profile
    """
    def __init__(self,
                domain,
                U,
                dU,
                d2U):
        self.domain = domain
        self.U = U
        self.dU = dU
        self.d2U = d2U

class VelocityProfileInitializer:
    def initialize(format):
        return get_velocity_profile(format)

def get_velocity_profile(format):
    if format == 'Poiseuille':
        return initialize_Poiseullie_flow()
    elif format == 'Couette':
        return initialize_Couette_flow()
    elif format == 'Blasius':
        return initialize_Blasius_flow()
    elif format == 'Falkner-Skan':
        return initialize_Falkner_Skan_flow()
    else:
        raise ValueError(format)

# Flow profile initializer functions

def initialize_Poiseullie_flow():
    domain = (-1,1)
    U = lambda x: 1-x**2
    dU = lambda x: -2*x
    d2U = lambda x: -2+0*x
    return VelocityProfile(domain,U,dU,d2U)

def initialize_Couette_flow():
    domain = (-1,1)
    U = lambda x: 1+x
    dU = lambda x: 1+0*x
    d2U = lambda x: 0*x
    return VelocityProfile(domain,U,dU,d2U)

def initialize_Blasius_flow():
    def blasius_rhs(y, x):
        return np.array([y[1], y[2], -0.5*y[0]*y[2]])

    def f(A, x_end):
        y0, x = [0, 0, A], np.linspace(0,x_end)
        sol = odeint(blasius_rhs, y0, x)
        return sol[-1,1]-1

    A0, x_end = 0.31, 40
    A = fsolve(f, A0, full_output=False, args=(x_end))

    y0, x = [0, 0, A], np.linspace(0,x_end)
    sol = odeint(blasius_rhs, y0, x)

    delta = trapz(1-sol[:,1],x)

    x = x/delta
    domain = (0,x_end/delta)
    U = interp1d(x,sol[:,1],kind='cubic')
    dU = interp1d(x,sol[:,2]*delta,kind='cubic')
    d2U = interp1d(x,-0.5*np.multiply(sol[:,0],sol[:,2])*delta**2,kind='cubic')

    return VelocityProfile(domain,U,dU,d2U)

def initialize_Falkner_Skan_flow():
    pass

def initialize_Bickley_jet_flow():
    pass

def initialize_cylinder_wake_flow():
    pass

# Velocity profile solver