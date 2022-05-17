import numpy as np


def speedgrad(x, u, I_, gamma=0.1):
    """
    This is a specific method for this system

    J = (I1-8)**2/2

    udot = -gamma * (I1-I_) *sin(theta2-theta1)
    """
    dt = 0.1
    udot = gamma * (x[0] - I_) * np.sin(x[3] - x[2])

    u_new = u + udot * dt
    return u_new
