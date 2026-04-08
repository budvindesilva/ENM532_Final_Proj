#Author 
#Budvin De Silva (University of Pennsylvania)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Values From Julian Forster (ETH Zurich)
# System Identication of the Crazy Flie 2.0 Nano Quadrocopter

m    = 0.025        # kg
Ixx  = 1.657171e-5  # kg·m²
Iyy  = 1.657171e-5  # kg·m²
Izz  = 2.926165e-5  # kg·m²
d    = 0.046        # m (arm length)
cT   = 1.28192e-8   # N·s²/rad²
cQ   = 7.6517e-11   # N·m·s²/rad²
g    = 9.81         # m/s²



def quadrotor_ode(t, state, inputs):
    x, y, z, xd, yd, zd, phi, theta, psi, p, q, r = state
    T, tau1, tau2, tau3 = inputs
    
    # phi, theta, psi are Euler angles (inertial frame, ZYX convention)
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    tth = np.tan(theta)
    
    #The rate of the Euler Angles
    #The kinematic equations relating body rates to Euler angle rates follow Stevens, Lewis & Johnson (2015), equation (1.4-4)
    phid = p + q * (sphi * tth)  + r*cphi*tth #phi_dot
    thetad = q * cphi - r *sphi  #theta_dot
    psid = q*sphi/cth + r*cphi/cth #psi_dot 
    
    #Body Rate derivatives from Mahony
    #Remember regular p, q, r are angular velocites from the body frame
    pdot = ((Iyy - Izz)/Ixx)*q*r + tau1/Ixx
    qdot = ((Izz - Ixx)/Iyy)*p*r + tau2/Iyy
    rdot = ((Ixx - Iyy)/Izz)*q*p + tau3/Izz

    
    #How fast the drone is accelerating among each axis
    xdd = (T/m) * (cpsi*sth + cth*sphi*spsi)     #acceleration in the x (intertial reference frame Z-X-Y)
    ydd = (T/m) *(spsi*sth - cpsi*cth*sphi)      #acceleration in the y (intertial reference frame Z-X-Y)
    zdd = (T/m) * (cphi*cth) - g                 #acceleration in the z (intertial reference frame Z-X-Y)
        
    return [xd, yd, zd,           # derivatives of position
        xdd, ydd, zdd,            # derivatives of velocity
        phid, thetad, psid,       # derivatives of Euler angles
        pdot, qdot, rdot]         # derivatives of body rates

# Initial Conditions
state0 = [0, 0, 0,      #position
          0, 0, 0,      #velocity
          0, 0, 0,      #euler angles
          0, 0, 0]      #angular velocity in the body frame

# Hover Thrust 
T_hover = m * g * 1.1
inputs = [T_hover, 0, 0, 0] #Tsigma, tau_1, tau_2, tau_3 

           
t_span = (0, 5)
t_eval = np.linspace(0, 5, 1000)

sol = solve_ivp(quadrotor_ode, t_span, state0,
                args=(inputs,), t_eval=t_eval,
                method='RK45', rtol=1e-8, atol=1e-10)

plt.plot(sol.t, sol.y[2])
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.title('Quadrotor z position (hover)')
plt.grid(True)
plt.show()